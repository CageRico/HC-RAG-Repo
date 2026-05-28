"""
Train the query-aware fusion gate used for paper-aligned HC-RAG inference.

The paper defines lambda as a learned routing weight over query features and
intent probabilities. The released dataset exposes the intent label and
structural text/table requirements, so we supervise the gate with a weak target
derived from those released annotations.
"""

import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import TextEncoder, TableEncoder, load_alignment_checkpoint
from src.fusion import AdaptiveFusionNetwork, compute_weak_lambda_target


INTENT_TO_ID = {
    "calculation": 0,
    "trend": 1,
    "fact": 2,
    "comparison": 3,
}


def _load_split(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _build_features(samples: List[Dict],
                    text_encoder: TextEncoder,
                    num_intents: int,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    queries = [str(s.get("question", "")).strip() for s in samples]
    query_emb_batches = []
    batch_size = 64
    with torch.no_grad():
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i + batch_size]
            query_emb_batches.append(text_encoder(batch).detach().cpu())
    query_embs = torch.cat(query_emb_batches, dim=0)

    intent_probs = []
    targets = []
    for sample in samples:
        intent = str(sample.get("intent", "fact")).lower()
        idx = INTENT_TO_ID.get(intent, INTENT_TO_ID["fact"])
        probs = torch.zeros(num_intents, dtype=torch.float32)
        probs[idx] = 1.0
        intent_probs.append(probs)
        targets.append(compute_weak_lambda_target(
            intent=intent,
            is_hybrid_modal=bool(sample.get("is_hybrid_modal", False)),
            subset=str(sample.get("subset", "")),
        ))

    return (
        query_embs.to(torch.float32),
        torch.stack(intent_probs, dim=0),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(1),
    )


def train(config_path: str = "config.yaml") -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_files_only = config["models"].get("local_files_only", True)
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    text_encoder = TextEncoder(
        model_name=config["models"]["text_encoder"],
        embedding_dim=config["alignment"]["embedding_dim"],
        local_files_only=local_files_only,
    )
    table_encoder = TableEncoder(
        model_name=config["models"]["table_encoder"],
        embedding_dim=config["alignment"]["embedding_dim"],
        local_files_only=local_files_only,
    )
    align_ckpt = os.path.join(checkpoint_dir, "align_checkpoint_best.pt")
    if load_alignment_checkpoint(text_encoder, table_encoder, align_ckpt, map_location=device):
        print(f"Loaded alignment checkpoint from {align_ckpt}")
    else:
        print("Alignment checkpoint not found; fusion training will use base encoder projections.")

    text_encoder.to(device).eval()

    train_path = os.path.join(config["paths"]["data_dir"], "multidoc2025", "train.json")
    val_path = os.path.join(config["paths"]["data_dir"], "multidoc2025", "val.json")
    train_samples = _load_split(train_path)
    val_samples = _load_split(val_path)

    num_intents = int(config["fusion"]["intent_classes"])
    q_train, p_train, y_train = _build_features(train_samples, text_encoder, num_intents, device)
    q_val, p_val, y_val = _build_features(val_samples, text_encoder, num_intents, device)

    train_ds = TensorDataset(q_train, p_train, y_train)
    val_ds = TensorDataset(q_val, p_val, y_val)
    batch_size = int(config["fusion"].get("batch_size", 64))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    fusion_net = AdaptiveFusionNetwork(
        embedding_dim=config["alignment"]["embedding_dim"],
        hidden_dim=config["fusion"]["hidden_dim"],
        num_intents=num_intents,
    ).to(device)

    optimizer = optim.Adam(
        fusion_net.gate.parameters(),
        lr=float(config["fusion"].get("learning_rate", 1e-4)),
    )
    criterion = nn.SmoothL1Loss()
    epochs = int(config["fusion"].get("epochs", 30))
    best_val = float("inf")

    print(f"Training fusion gate for {epochs} epochs on {len(train_samples)} samples")
    for epoch in range(epochs):
        fusion_net.train()
        train_loss = 0.0
        for query_emb, intent_probs, targets in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            query_emb = query_emb.to(device)
            intent_probs = intent_probs.to(device)
            targets = targets.to(device)

            pred = fusion_net.gate(torch.cat([query_emb, intent_probs], dim=-1))
            loss = criterion(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        fusion_net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for query_emb, intent_probs, targets in val_loader:
                query_emb = query_emb.to(device)
                intent_probs = intent_probs.to(device)
                targets = targets.to(device)
                pred = fusion_net.gate(torch.cat([query_emb, intent_probs], dim=-1))
                val_loss += criterion(pred, targets).item()

        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        print(f"Epoch {epoch + 1}/{epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        payload = {
            "epoch": epoch + 1,
            "model_state_dict": fusion_net.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, os.path.join(checkpoint_dir, "fusion_best.pt"))
            print("  Saved best -> fusion_best.pt")

    torch.save(payload, os.path.join(checkpoint_dir, "fusion_final.pt"))
    print(f"Training complete. Best validation loss: {best_val:.4f}")


if __name__ == "__main__":
    train()
