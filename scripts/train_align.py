"""
Cross-modal contrastive alignment training
Aligns text and table representations in shared space

Optimization: pre-compute all embeddings once, cache to disk,
then train only the projection heads with cached embeddings.
This eliminates repeated TAPAS/FinBERT inference during training.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
from tqdm import tqdm
import yaml
import os
import sys
import pickle
from typing import List, Tuple, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import TextEncoder, TableEncoder, CrossModalAligner, contrastive_loss


# ---------------------------------------------------------------------------
# Dataset that works directly with pre-computed embedding tensors
# ---------------------------------------------------------------------------

class EmbeddingPairDataset(Dataset):
    def __init__(self, text_embs: torch.Tensor, table_embs: torch.Tensor):
        assert len(text_embs) == len(table_embs)
        self.text_embs = text_embs
        self.table_embs = table_embs

    def __len__(self):
        return len(self.text_embs)

    def __getitem__(self, idx):
        return self.text_embs[idx], self.table_embs[idx]


# ---------------------------------------------------------------------------
# Lightweight projection-only aligner (used after pre-computation)
# ---------------------------------------------------------------------------

class ProjectionAligner(nn.Module):
    """
    Trains only the projection heads on top of frozen base embeddings.
    Much faster than full encoder forward passes each batch.
    """
    def __init__(self, embedding_dim: int = 768, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
        self.text_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        self.table_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, text_embs: torch.Tensor, table_embs: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F
        t = F.normalize(self.text_proj(text_embs), p=2, dim=1)
        v = F.normalize(self.table_proj(table_embs), p=2, dim=1)
        logits = torch.matmul(t, v.T) / self.temperature
        labels = torch.arange(len(t), device=t.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        return loss


# ---------------------------------------------------------------------------
# Pre-computation helpers
# ---------------------------------------------------------------------------

def precompute_embeddings(
    pairs: List[Tuple[str, Dict]],
    text_encoder: TextEncoder,
    table_encoder: TableEncoder,
    cache_path: str,
    batch_size: int = 64,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode all texts and tables once, save to cache_path.
    Returns (text_embs, table_embs) as numpy arrays of shape (N, D).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    text_encoder.to(device).eval()
    table_encoder.to(device).eval()

    texts  = [p[0] for p in pairs]
    tables = [p[1] for p in pairs]

    print(f"Pre-computing {len(texts)} text embeddings ...")
    text_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Text encoding"):
        batch = texts[i: i + batch_size]
        with torch.no_grad():
            emb = text_encoder(batch)
        text_embs.append(emb.cpu().numpy())
    text_embs = np.concatenate(text_embs, axis=0)

    print(f"Pre-computing {len(tables)} table embeddings ...")
    table_embs = []
    for i in tqdm(range(0, len(tables), 1), desc="Table encoding"):  # TAPAS: 1 at a time
        with torch.no_grad():
            emb = table_encoder.encode_table(tables[i])
        table_embs.append(emb.cpu().numpy())
    table_embs = np.concatenate(table_embs, axis=0)

    np.save(cache_path + "_text.npy", text_embs)
    np.save(cache_path + "_table.npy", table_embs)
    print(f"Saved embedding cache to {cache_path}_{{text,table}}.npy")
    return text_embs, table_embs


def load_or_compute_embeddings(
    pairs: List[Tuple[str, Dict]],
    text_encoder: TextEncoder,
    table_encoder: TableEncoder,
    cache_path: str,
    batch_size: int = 64,
    device: torch.device = None,
) -> Tuple[np.ndarray, np.ndarray]:
    text_cache  = cache_path + "_text.npy"
    table_cache = cache_path + "_table.npy"
    if os.path.exists(text_cache) and os.path.exists(table_cache):
        print(f"Loading cached embeddings from {cache_path}_{{text,table}}.npy ...")
        text_embs  = np.load(text_cache)
        table_embs = np.load(table_cache)
        if len(text_embs) == len(pairs):
            print(f"Cache hit: {len(text_embs)} pairs")
            return text_embs, table_embs
        print("Cache size mismatch, recomputing ...")
    return precompute_embeddings(pairs, text_encoder, table_encoder,
                                 cache_path, batch_size, device)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class AlignmentTrainer:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.alignment_config = self.config["alignment"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_gpus = torch.cuda.device_count()
        print(f"Using device: {self.device}, GPUs available: {self.n_gpus}")

        self.text_encoder = TextEncoder(
            model_name=self.config["models"]["text_encoder"],
            embedding_dim=self.alignment_config["embedding_dim"],
            local_files_only=self.config["models"].get("local_files_only", True),
        )
        self.table_encoder = TableEncoder(
            model_name=self.config["models"]["table_encoder"],
            embedding_dim=self.alignment_config["embedding_dim"],
            local_files_only=self.config["models"].get("local_files_only", True),
        )

        self.checkpoint_dir = self.config["paths"]["checkpoint_dir"]
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train(self, num_epochs: int, pairs: List[Tuple[str, Dict]]):
        cache_path = os.path.join(self.checkpoint_dir, "align_emb_cache")

        # Step 1: pre-compute embeddings (multi-GPU for text encoder)
        # Wrap text encoder with DataParallel for faster batch encoding
        if self.n_gpus > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)
        self.text_encoder.to(self.device)

        text_embs, table_embs = load_or_compute_embeddings(
            pairs,
            self.text_encoder.module if self.n_gpus > 1 else self.text_encoder,
            self.table_encoder,
            cache_path,
            batch_size=self.alignment_config.get("batch_size", 64) * max(1, self.n_gpus),
            device=self.device,
        )

        # Step 2: train projection aligner on cached embeddings
        dim = self.alignment_config["embedding_dim"]
        tau = float(self.alignment_config["temperature"])
        aligner = ProjectionAligner(embedding_dim=dim, temperature=tau)
        if self.n_gpus > 1:
            aligner = nn.DataParallel(aligner)
        aligner = aligner.to(self.device)

        optimizer = optim.Adam(aligner.parameters(),
                               lr=float(self.alignment_config["learning_rate"]))

        text_t  = torch.tensor(text_embs,  dtype=torch.float32)
        table_t = torch.tensor(table_embs, dtype=torch.float32)
        dataset = EmbeddingPairDataset(text_t, table_t)
        batch_size = min(
            self.alignment_config.get("batch_size", 128) * max(1, self.n_gpus),
            len(dataset)
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        print(f"\nTraining projection aligner for {num_epochs} epochs "
              f"({len(dataset)} pairs, batch={batch_size}, gpus={self.n_gpus}) ...")
        best_loss = float("inf")
        for epoch in range(num_epochs):
            aligner.train()
            total_loss = 0.0
            for text_b, table_b in tqdm(loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                text_b  = text_b.to(self.device)
                table_b = table_b.to(self.device)
                loss = aligner(text_b, table_b)
                if self.n_gpus > 1:
                    loss = loss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg = total_loss / len(loader)
            print(f"Epoch {epoch+1}/{num_epochs}  loss={avg:.4f}")

            # Save (unwrap DataParallel if needed)
            model_to_save = aligner.module if self.n_gpus > 1 else aligner
            if (epoch + 1) % 5 == 0:
                self._save(model_to_save, optimizer, epoch, avg,
                           os.path.join(self.checkpoint_dir, f"align_checkpoint_{epoch+1}.pt"))
            if avg < best_loss:
                best_loss = avg
                self._save(model_to_save, optimizer, epoch, avg,
                           os.path.join(self.checkpoint_dir, "align_checkpoint_best.pt"))

        self._save(model_to_save, optimizer, num_epochs, avg,
                   os.path.join(self.checkpoint_dir, "align_checkpoint_final.pt"))
        print(f"Training complete. Best loss: {best_loss:.4f}")

    def _save(self, model, optimizer, epoch, loss, path):
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
        }, path)
        print(f"  Saved checkpoint → {path}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_pairs_from_index(index_path: str, max_pairs: int = 50000) -> List[Tuple[str, Dict]]:
    from src.hierarchical_index import HierarchicalIndex, NodeType, EdgeType

    print(f"Loading index from {index_path} ...")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    index = HierarchicalIndex(cfg["index"])
    index.load(index_path)

    parent_map: Dict[str, str] = {}
    for child_id, edges in index.reverse_edges.items():
        for parent_id, edge_type in edges:
            if edge_type == EdgeType.HIERARCHICAL:
                parent_map[child_id] = parent_id
                break

    section_texts:  Dict[str, List[str]]  = {}
    section_tables: Dict[str, List[Dict]] = {}

    for node_id, node in index.nodes.items():
        if node.node_type == NodeType.TEXT_CHUNK:
            parent_id = parent_map.get(node_id)
            if parent_id and isinstance(node.content, str) and node.content.strip():
                section_texts.setdefault(parent_id, []).append(node.content)
        elif node.node_type == NodeType.TABLE_CELL:
            parent_id = parent_map.get(node_id)
            if parent_id and isinstance(node.content, dict):
                c = node.content
                section_tables.setdefault(parent_id, []).append({
                    "header": [c.get("col_header", "")],
                    "rows":   [[c.get("row_header", ""), c.get("value", "")]]
                })

    pairs = []
    for section_id in section_texts:
        if section_id not in section_tables:
            continue
        texts  = section_texts[section_id]
        tables = section_tables[section_id]
        for i, text in enumerate(texts):
            pairs.append((text, tables[i % len(tables)]))
            if len(pairs) >= max_pairs:
                return pairs

    print(f"Loaded {len(pairs)} real text-table pairs from index")
    return pairs


def create_synthetic_pairs():
    examples = [
        ("Revenue increased by 15% to $100M in Q4 2024",
         {"header": ["Quarter", "Revenue"], "rows": [["Q4 2023", "$87M"], ["Q4 2024", "$100M"]]}),
        ("Gross margin improved to 45% from 42% last year",
         {"header": ["Year", "Gross Margin"], "rows": [["2023", "42%"], ["2024", "45%"]]}),
        ("Operating expenses were $30M, up 10% year over year",
         {"header": ["Metric", "2023", "2024"], "rows": [["Operating Expenses", "$27.3M", "$30M"]]}),
    ]
    return examples * 20


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = AlignmentTrainer("config.yaml")

    index_path = "./indexes/hierarchical_index.pkl"
    if os.path.exists(index_path):
        pairs = load_pairs_from_index(index_path)
        if len(pairs) < 10:
            print("Too few pairs from index, falling back to synthetic data")
            pairs = create_synthetic_pairs()
    else:
        print("Index not found, using synthetic data")
        pairs = create_synthetic_pairs()

    print(f"Total pairs: {len(pairs)}")
    num_epochs = int(trainer.alignment_config.get("epochs", 15))
    trainer.train(num_epochs=num_epochs, pairs=pairs)
