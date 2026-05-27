"""
Intent classifier training v2 — fine-tune FinBERT end-to-end.
Instead of training an MLP on frozen embeddings, we fine-tune the full
FinBERT model with a classification head. This gives much better accuracy
on small datasets (1600 samples) because the model can adapt its
representations to the intent classification task.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm
import yaml
import os
import sys
import json
from typing import List, Tuple, Dict
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class IntentDataset(Dataset):
    def __init__(self, samples: List[Tuple[str, int]], tokenizer, max_length: int = 128):
        self.samples = samples
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, label = self.samples[idx]
        enc = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          torch.tensor(label, dtype=torch.long),
        }


class FinBERTIntentClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 4, dropout: float = 0.2,
                 local_files_only: bool = True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(cls)


def create_training_data(
    multidoc_path: str = "./data/multidoc2025/train.json",
) -> Tuple[List, List]:
    intent_map = {"calculation": 0, "trend": 1, "fact": 2, "comparison": 3}
    train_samples, val_samples = [], []

    if os.path.exists(multidoc_path):
        with open(multidoc_path, encoding="utf-8") as f:
            for row in json.load(f):
                intent = row.get("intent", "fact")
                if intent in intent_map:
                    train_samples.append((row["question"], intent_map[intent]))
        print(f"  Train: {len(train_samples)} samples from Multi-Doc-2025")

    val_path = multidoc_path.replace("train.json", "val.json")
    if os.path.exists(val_path):
        with open(val_path, encoding="utf-8") as f:
            for row in json.load(f):
                intent = row.get("intent", "fact")
                if intent in intent_map:
                    val_samples.append((row["question"], intent_map[intent]))
        print(f"  Val:   {len(val_samples)} samples from Multi-Doc-2025")

    from collections import Counter
    print(f"  Train dist: {dict(Counter(l for _, l in train_samples))}")
    print(f"  Val   dist: {dict(Counter(l for _, l in val_samples))}")
    return train_samples, val_samples


def train():
    with open("config.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpus = torch.cuda.device_count()
    print(f"Device: {device}, GPUs: {n_gpus}")

    model_name   = config["models"]["text_encoder"]
    local_files_only = config["models"].get("local_files_only", True)
    num_classes  = config["fusion"]["intent_classes"]
    dropout      = config["fusion"]["dropout"]
    checkpoint_dir = config["paths"]["checkpoint_dir"]
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Hyperparams tuned for small dataset fine-tuning
    EPOCHS     = 20
    BATCH_SIZE = 32
    LR         = 2e-5       # standard BERT fine-tuning LR
    WARMUP     = 0.1        # 10% warmup
    MAX_LEN    = 128
    WEIGHT_DECAY = 0.01

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    train_data, val_data = create_training_data()

    train_ds = IntentDataset(train_data, tokenizer, MAX_LEN)
    val_ds   = IntentDataset(val_data,   tokenizer, MAX_LEN)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = FinBERTIntentClassifier(model_name, num_classes, dropout, local_files_only)
    if n_gpus > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * WARMUP),
        num_training_steps=total_steps,
    )

    best_acc = 0.0
    for epoch in range(EPOCHS):
        # Train
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lbls = batch["label"].to(device)
            logits = model(ids, mask)
            loss = criterion(logits, lbls)
            if n_gpus > 1:
                loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        # Validate
        model.eval()
        correct = total = val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                lbls = batch["label"].to(device)
                logits = model(ids, mask)
                loss = criterion(logits, lbls)
                if n_gpus > 1:
                    loss = loss.mean()
                val_loss += loss.item()
                _, pred = torch.max(logits, 1)
                total   += lbls.size(0)
                correct += (pred == lbls).sum().item()

        val_acc = 100 * correct / total
        print(f"Epoch {epoch+1}/{EPOCHS}  "
              f"train_loss={total_loss/len(train_loader):.4f}  "
              f"val_loss={val_loss/len(val_loader):.4f}  "
              f"val_acc={val_acc:.2f}%")

        m = model.module if n_gpus > 1 else model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({"model_state_dict": m.state_dict(), "accuracy": val_acc},
                       os.path.join(checkpoint_dir, "intent_best.pt"))
            print(f"  Saved best → intent_best.pt")

    torch.save({"model_state_dict": m.state_dict(), "accuracy": best_acc},
               os.path.join(checkpoint_dir, "intent_final.pt"))
    print(f"Best validation accuracy: {best_acc:.2f}%")


if __name__ == "__main__":
    train()
