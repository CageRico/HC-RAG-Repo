"""
Cross-Modal Dual-Stream Encoding with Contrastive Alignment
- Text Encoder: FinBERT with domain adaptation
- Table Encoder: TAPAS with financial table understanding
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass


class TextEncoder(nn.Module):
    """
    Financial-domain-adapted text encoder
    Based on FinBERT with contrastive fine-tuning
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", 
                 embedding_dim: int = 768,
                 max_length: int = 512,
                 local_files_only: bool = True):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)

        # Projection head to shared embedding space
        self.projection = nn.Sequential(
            nn.Linear(768, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
        
    def forward(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings"""
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to same device as model
        inputs = {k: v.to(self.encoder.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings = self.projection(cls_embeddings)
            return F.normalize(embeddings, p=2, dim=1)
    
    def encode_query(self, query: str) -> torch.Tensor:
        """Encode a single query"""
        return self.forward([query])
    
    def to_device(self, device: torch.device):
        self.to(device)
        return self


class TableEncoder(nn.Module):
    """
    TAPAS-based table encoder.
    TAPAS is pre-trained on Wikipedia tables for table question answering,
    providing strong structural understanding of financial tables.
    Reference: Herzig et al., 2020 (google/tapas-large-finetuned-wtq)
    """

    def __init__(self, model_name: str = "google/tapas-large-finetuned-wtq",
                 embedding_dim: int = 768,
                 max_cells: int = 500,
                 local_files_only: bool = True):
        super().__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.max_cells = max_cells

        self.encoder   = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)

        hidden_size = self.encoder.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def _to_pandas(self, table_data: Dict[str, Any]):
        """Convert table dict to pandas DataFrame for TapasTokenizer."""
        import pandas as pd
        header = table_data.get("header", [])
        rows   = table_data.get("rows",   [])
        if not header:
            header = [f"Col{i}" for i in range(len(rows[0]) if rows else 1)]
        # Truncate rows to max_cells / num_cols
        n_cols   = max(len(header), 1)
        max_rows = max(1, self.max_cells // n_cols)
        rows     = rows[:max_rows]
        # Pad/trim each row to match header length
        rows = [(r + [""] * n_cols)[:n_cols] for r in rows]
        return pd.DataFrame(rows, columns=[str(h) for h in header])

    def encode_table(self, table: Dict[str, Any], question: str = "") -> torch.Tensor:
        """Encode a table dict to a normalized embedding vector."""
        df = self._to_pandas(table)
        if not question:
            question = "Summarize the financial data in this table."
        inputs = self.tokenizer(
            table=df,
            queries=question,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        inputs = {k: v.to(next(self.encoder.parameters()).device)
                  for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            # CLS token as table representation
            embedding = outputs.last_hidden_state[:, 0, :]
        embedding = self.projection(embedding)
        return F.normalize(embedding, p=2, dim=1)

    def encode_tables_batch(self, tables: List[Dict[str, Any]],
                            questions: List[str] = None) -> torch.Tensor:
        if questions is None:
            questions = [""] * len(tables)
        return torch.cat([self.encode_table(t, q)
                          for t, q in zip(tables, questions)], dim=0)

    def to_device(self, device: torch.device):
        self.to(device)
        return self


class CrossModalAligner(nn.Module):
    """
    Contrastive learning for cross-modal alignment
    Aligns text and table representations in shared space
    """
    
    def __init__(self, text_encoder: TextEncoder, 
                 table_encoder: TableEncoder,
                 embedding_dim: int = 768,
                 temperature: float = 0.07):
        super().__init__()
        self.text_encoder = text_encoder
        self.table_encoder = table_encoder
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        
    def forward(self, texts: List[str], tables: List[Dict[str, Any]]) -> torch.Tensor:
        """
        Compute contrastive loss between text and table pairs
        """
        text_embeds = self.text_encoder(texts)
        table_embeds = self.table_encoder.encode_tables_batch(tables)
        
        # Compute similarity matrix
        similarity = torch.matmul(text_embeds, table_embeds.T) / self.temperature
        
        # Contrastive loss (InfoNCE)
        labels = torch.arange(len(texts)).to(similarity.device)
        loss_text = F.cross_entropy(similarity, labels)
        loss_table = F.cross_entropy(similarity.T, labels)
        
        return (loss_text + loss_table) / 2
    
    def compute_similarity(self, text: str, table: Dict[str, Any]) -> float:
        """Compute similarity between text and table"""
        text_emb = self.text_encoder([text])
        table_emb = self.table_encoder.encode_table(table)
        return torch.dot(text_emb[0], table_emb[0]).item()
    
    def get_aligned_embeddings(self, texts: List[str], tables: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get aligned embeddings for both modalities"""
        text_embeds = self.text_encoder(texts)
        table_embeds = self.table_encoder.encode_tables_batch(tables)
        return text_embeds, table_embeds


def contrastive_loss(text_embeds: torch.Tensor, 
                     table_embeds: torch.Tensor, 
                     temperature: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss for cross-modal alignment
    """
    # Normalize embeddings
    text_embeds = F.normalize(text_embeds, p=2, dim=1)
    table_embeds = F.normalize(table_embeds, p=2, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(text_embeds, table_embeds.T) / temperature
    
    # Labels are diagonal (matched pairs)
    labels = torch.arange(len(text_embeds)).to(logits.device)
    
    # Symmetric loss
    loss_text = F.cross_entropy(logits, labels)
    loss_table = F.cross_entropy(logits.T, labels)
    
    return (loss_text + loss_table) / 2


class RetrievalEncoder:
    """
    Wrapper for encoding queries and documents for retrieval
    """

    BATCH_SIZE = 64  # tune based on GPU VRAM

    def __init__(self, text_encoder: TextEncoder, table_encoder: TableEncoder):
        self.text_encoder = text_encoder
        self.table_encoder = table_encoder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Move encoders to device once at init
        self.text_encoder.to_device(self.device)
        self.text_encoder.eval()
        self.table_encoder.to_device(self.device)
        self.table_encoder.eval()

    def encode_query(self, query: str) -> np.ndarray:
        with torch.no_grad():
            embedding = self.text_encoder.encode_query(query)
        return embedding.cpu().numpy()[0]

    def encode_text_chunk(self, text: str) -> np.ndarray:
        with torch.no_grad():
            embedding = self.text_encoder([text])
        return embedding.cpu().numpy()[0]

    def encode_text_chunks_batch(self, texts: List[str]) -> np.ndarray:
        """Batch-encode a list of text chunks; returns (N, D) numpy array."""
        results = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i: i + self.BATCH_SIZE]
            with torch.no_grad():
                emb = self.text_encoder(batch)
            results.append(emb.cpu().numpy())
        return np.concatenate(results, axis=0) if results else np.empty((0, self.text_encoder.embedding_dim))

    def encode_table_cell(self, cell_content: Dict[str, Any]) -> np.ndarray:
        cell_text = f"{cell_content.get('row_header', '')} {cell_content.get('col_header', '')} {cell_content.get('value', '')}"
        return self.encode_text_chunk(cell_text.strip())

    def encode_table_cells_batch(self, cell_texts: List[str]) -> np.ndarray:
        """Batch-encode a list of pre-formatted cell text strings."""
        return self.encode_text_chunks_batch(cell_texts)
