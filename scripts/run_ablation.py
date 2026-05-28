"""
Ablation study for HC-RAG (Table 4 in the paper).

Eight variants evaluated on Multi-Doc-2025 test set:
  full          -HC-RAG (Full)
  no_hierarchy  -w/o Three-Level Index  (flat dense retrieval over all chunks)
  no_alignment  -w/o Cross-Modal Alignment  (skip loading align checkpoint)
  no_tapas      -w/o TAPAS  (use FinBERT for table encoding too)
  no_tapex      -alias for no_tapas kept for backward compatibility
  no_fusion     -w/o Query-Aware Fusion  (fixed lambda = 0.5)
  no_l1_edges   -w/o L1 Cross-Doc Edges  (L1 returns only top-1 doc)
  no_l2_section -w/o L2 Section Nodes  (skip L2, go L1->L3 directly)
  no_l3_table   -w/o L3 Table Structure  (ignore table cells, text-only L3)

Usage:
  python scripts/run_ablation.py                          # all variants, multidoc2025
  python scripts/run_ablation.py --variant no_hierarchy   # single variant
  python scripts/run_ablation.py --max_samples 100        # quick test
"""

import os
import sys
import json
import time
import csv
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_index import HierarchicalIndex, NodeType, BaseNode, DocumentNode, SectionNode
from src.encoders import TextEncoder, TableEncoder, RetrievalEncoder, load_alignment_checkpoint
from src.fusion import AdaptiveFusionNetwork, IntentType, load_fusion_checkpoint
from src.retriever import HierarchicalRetriever, ContextBuilder
from src.generator import ResponseGenerator
from src.evaluation import BenchmarkEvaluator
from scripts.prepare_datasets import load_dataset_split
from scripts.run_inference import FinBERTIntentClassifier, FinBERTIntentWrapper
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Flat retriever -used by no_hierarchy variant
# ---------------------------------------------------------------------------

class FlatRetriever:
    """
    Flat dense retrieval over all L3 chunks (no L1/L2 filtering).
    Replaces HierarchicalRetriever for the w/o Three-Level Index ablation.
    """
    def __init__(self, index: HierarchicalIndex, encoder: RetrievalEncoder,
                 fusion_network: AdaptiveFusionNetwork, intent_classifier,
                 config: Dict, fixed_lambda: float = None):
        self.index = index
        self.encoder = encoder
        self.fusion_network = fusion_network
        self.intent_classifier = intent_classifier
        self.l3_k = config.get("l3_semantic_k", 20)
        self.fixed_lambda = fixed_lambda

    def retrieve(self, query: str) -> Tuple[List[BaseNode], float, IntentType]:
        query_emb = self.encoder.encode_query(query)

        # Intent
        _emb = torch.from_numpy(query_emb).float()
        try:
            intent_probs = self.intent_classifier.get_intent_probs(_emb, query=query)
            intent = self.intent_classifier.predict_intent(_emb, query=query)
        except TypeError:
            intent_probs = self.intent_classifier.get_intent_probs(_emb)
            intent = self.intent_classifier.predict_intent(_emb)

        # Collect ALL L3 nodes (no L1/L2 filtering)
        text_chunks, table_cells = [], []
        for node_id, node in self.index.nodes.items():
            if node.node_type == NodeType.TEXT_CHUNK:
                text_chunks.append(node)
            elif node.node_type == NodeType.TABLE_CELL:
                table_cells.append(node)

        def _score(nodes, use_text=True):
            scored = []
            for node in nodes:
                emb = getattr(node, "embedding", None)
                if emb is None:
                    if use_text:
                        c = node.content if isinstance(node.content, str) else ""
                        emb = self.encoder.encode_text_chunk(c[:512])
                    else:
                        c = node.content if isinstance(node.content, dict) else {}
                        txt = f"{c.get('row_header','')} {c.get('col_header','')} {c.get('value','')}"
                        emb = self.encoder.encode_text_chunk(txt)
                    node.embedding = emb
                sim = float(np.dot(emb, query_emb) /
                            (np.linalg.norm(emb) * np.linalg.norm(query_emb) + 1e-9))
                scored.append((node, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored

        text_scored  = _score(text_chunks,  use_text=True)
        table_scored = _score(table_cells, use_text=False)

        # Fusion weight
        if self.fixed_lambda is not None:
            lam = self.fixed_lambda
        else:
            device = next(self.fusion_network.gate.parameters()).device
            qt = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)
            it = torch.from_numpy(intent_probs).float().unsqueeze(0).to(device)
            with torch.no_grad():
                lam = self.fusion_network.gate(torch.cat([qt, it], dim=-1)).item()

        merged = ([(n, s * lam,       "text")  for n, s in text_scored[:self.l3_k]] +
                  [(n, s * (1 - lam), "table") for n, s in table_scored[:self.l3_k]])
        merged.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in merged[:self.l3_k]], lam, intent


# ---------------------------------------------------------------------------
# Ablation variant factory
# ---------------------------------------------------------------------------

def build_variant(variant: str, config: dict, device: torch.device):
    """
    Build (retriever, generator) pair for the given ablation variant.
    Returns (retriever, context_builder, generator).
    """
    cfg_models    = config["models"]
    cfg_align     = config["alignment"]
    cfg_fusion    = config["fusion"]
    cfg_paths     = config["paths"]
    cfg_index_cfg = config["index"]
    local_files_only = cfg_models.get("local_files_only", True)

    # ---- Text encoder (shared by all variants) ----
    text_enc = TextEncoder(model_name=cfg_models["text_encoder"],
                           embedding_dim=cfg_align["embedding_dim"],
                           local_files_only=local_files_only)
    text_enc.to(device).eval()

    variant = "no_tapas" if variant == "no_tapex" else variant

    # ---- Table encoder ----
    if variant == "no_tapas":
        # Replace TAPAS with a second FinBERT instance
        table_enc = TextEncoder(model_name=cfg_models["text_encoder"],
                                embedding_dim=cfg_align["embedding_dim"],
                                local_files_only=local_files_only)
        print("  [ablation] Table encoder: FinBERT (no TAPAS)")
    else:
        table_enc = TableEncoder(model_name=cfg_models["table_encoder"],
                                 embedding_dim=cfg_align["embedding_dim"],
                                 local_files_only=local_files_only)
    table_enc.to(device).eval()

    align_ckpt = os.path.join(cfg_paths["checkpoint_dir"], "align_checkpoint_best.pt")
    if variant == "no_alignment":
        print("  [ablation] Cross-modal alignment: DISABLED")
    elif variant == "no_tapas":
        print("  [ablation] Cross-modal alignment: skipped for no_tapas variant")
    elif load_alignment_checkpoint(text_enc, table_enc, align_ckpt, map_location=device):
        print(f"  [ablation] Alignment checkpoint: loaded from {align_ckpt}")
    else:
        print("  [ablation] Alignment checkpoint: not found")

    retrieval_enc = RetrievalEncoder(text_enc, table_enc)

    # ---- Intent classifier (shared) ----
    tok = AutoTokenizer.from_pretrained(cfg_models["text_encoder"], local_files_only=local_files_only)
    clf_model = FinBERTIntentClassifier(
        model_name=cfg_models["text_encoder"],
        num_classes=cfg_fusion["intent_classes"],
        dropout=cfg_fusion["dropout"],
        local_files_only=local_files_only,
    )
    ckpt = os.path.join(cfg_paths["checkpoint_dir"], "intent_best.pt")
    if os.path.exists(ckpt):
        clf_model.load_state_dict(
            torch.load(ckpt, map_location=device)["model_state_dict"])
    clf_model.to(device).eval()
    intent_clf = FinBERTIntentWrapper(clf_model, tok, device)

    # ---- Fusion network ----
    fusion_net = AdaptiveFusionNetwork(
        embedding_dim=cfg_align["embedding_dim"],
        hidden_dim=cfg_fusion["hidden_dim"],
        num_intents=cfg_fusion["intent_classes"],
    )
    if variant != "no_fusion":
        fusion_ckpt = os.path.join(cfg_paths["checkpoint_dir"], "fusion_best.pt")
        if load_fusion_checkpoint(fusion_net, fusion_ckpt, map_location=device):
            print(f"  [ablation] Fusion checkpoint: loaded from {fusion_ckpt}")
        else:
            print("  [ablation] Fusion checkpoint: not found")
    fusion_net.to(device).eval()

    # ---- Index ----
    index = HierarchicalIndex(cfg_index_cfg)
    index_path = os.path.join(cfg_paths["index_dir"], "hierarchical_index.pkl")
    if os.path.exists(index_path):
        index.load(index_path)

    # ---- Retriever ----
    if variant == "no_hierarchy":
        retriever = FlatRetriever(index, retrieval_enc, fusion_net, intent_clf, cfg_index_cfg)
        print("  [ablation] Retriever: FLAT (no L1/L2)")

    elif variant == "no_fusion":
        # Fixed lambda = 0.5 -use FlatRetriever with fixed_lambda but keep hierarchy
        # We patch HierarchicalRetriever by monkey-patching _retrieve_semantic_units
        retriever = HierarchicalRetriever(index, retrieval_enc, fusion_net,
                                          intent_clf, cfg_index_cfg)
        _patch_fixed_lambda(retriever, fixed_lambda=0.5)
        print("  [ablation] Fusion: FIXED lambda=0.5")

    elif variant == "no_l1_edges":
        # L1 returns only the single best-scoring doc (no cross-doc edge traversal)
        retriever = HierarchicalRetriever(index, retrieval_enc, fusion_net,
                                          intent_clf, cfg_index_cfg)
        retriever.l1_k = 1
        print("  [ablation] L1: top-1 doc only (no cross-doc edges)")

    elif variant == "no_l2_section":
        # Skip L2: after L1 doc selection, go directly to all L3 nodes in those docs
        retriever = HierarchicalRetriever(index, retrieval_enc, fusion_net,
                                          intent_clf, cfg_index_cfg)
        _patch_skip_l2(retriever)
        print("  [ablation] L2: SKIPPED (L1 ->L3 directly)")

    elif variant == "no_l3_table":
        # L3: text chunks only, ignore table cells
        retriever = HierarchicalRetriever(index, retrieval_enc, fusion_net,
                                          intent_clf, cfg_index_cfg)
        _patch_text_only_l3(retriever)
        print("  [ablation] L3: TEXT ONLY (no table cells)")

    else:
        # full / no_alignment / no_tapas -standard hierarchical retriever
        retriever = HierarchicalRetriever(index, retrieval_enc, fusion_net,
                                          intent_clf, cfg_index_cfg)
        print(f"  [ablation] Retriever: FULL hierarchical")

    # ---- Generator ----
    generator = ResponseGenerator({
        "model_name":      cfg_models["generator"],
        "max_tokens":      config["generation"]["max_tokens"],
        "temperature":     config["generation"]["temperature"],
        "openai_api_key":  cfg_models.get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", ""),
        "openai_base_url": cfg_models.get("openai_base_url", "") or os.getenv("OPENAI_BASE_URL", ""),
        "local_files_only": cfg_models.get("local_files_only", True),
    })

    return retriever, ContextBuilder(), generator


# ---------------------------------------------------------------------------
# Monkey-patches for retriever variants
# ---------------------------------------------------------------------------

def _patch_fixed_lambda(retriever: HierarchicalRetriever, fixed_lambda: float):
    """Replace adaptive gate with fixed lambda."""
    original = retriever._retrieve_semantic_units

    def _fixed(query, query_embedding, sections, intent_probs):
        nodes, _ = original(query, query_embedding, sections, intent_probs)
        return nodes, fixed_lambda

    retriever._retrieve_semantic_units = _fixed


def _patch_skip_l2(retriever: HierarchicalRetriever):
    """After L1 doc selection, collect all L3 nodes directly (skip L2 section scoring)."""
    def _no_l2(query, query_embedding, documents):
        # Return a dummy single "section" per doc so L3 still runs
        all_sections = []
        for doc in documents:
            all_sections.extend(retriever.index.get_document_sections(doc.node_id))
        return all_sections  # return all sections unfiltered

    retriever._retrieve_sections = _no_l2


def _patch_text_only_l3(retriever: HierarchicalRetriever):
    """L3: only score text chunks, skip table cells entirely."""
    original = retriever._retrieve_semantic_units

    def _text_only(query, query_embedding, sections, intent_probs):
        text_chunks = []
        for section in sections:
            for chunk in retriever.index.get_section_chunks(section.node_id):
                if chunk.node_type == NodeType.TEXT_CHUNK:
                    text_chunks.append(chunk)

        text_embeds = []
        for chunk in text_chunks:
            emb = getattr(chunk, "embedding", None)
            if emb is None:
                c = chunk.content if isinstance(chunk.content, str) else ""
                emb = retriever.encoder.encode_text_chunk(c[:512])
                chunk.embedding = emb
            text_embeds.append(emb)

        if not text_embeds:
            return [], 1.0

        text_embeds_np = np.stack(text_embeds)
        sims = np.dot(text_embeds_np, query_embedding)
        ranked = np.argsort(sims)[::-1][:retriever.l3_k]
        return [text_chunks[i] for i in ranked], 1.0  # lambda=1 (text only)

    retriever._retrieve_semantic_units = _text_only


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def _clean_sample(sample: dict) -> dict:
    q = sample.get("question", "")
    a = sample.get("answer", "")
    sample = dict(sample)
    if isinstance(q, str):
        for marker in ("Question:", "question:", "Q:"):
            idx = q.rfind(marker)
            if idx != -1:
                tail = q[idx + len(marker):].strip()
                for end in ("\nAnswer:", "\nA:", "\n"):
                    if end in tail:
                        tail = tail[:tail.index(end)].strip()
                        break
                if tail:
                    q = tail
                    break
    sample["question"] = str(q).strip()
    if isinstance(a, (list, dict)):
        sample["answer"] = ", ".join(str(x) for x in a) if isinstance(a, list) else str(a)
    else:
        sample["answer"] = str(a)
    return sample


def evaluate_variant(variant_name, retriever, context_builder, generator,
                     evaluator, samples, max_samples=None, workers=16):
    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    results = [None] * n
    latencies = [0.0] * n
    print_lock = threading.Lock()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _process(idx, raw):
        sample = _clean_sample(raw)
        t0 = time.perf_counter()
        try:
            evidence_nodes, fusion_weight, intent = retriever.retrieve(sample["question"])
            result = generator.generate(sample["question"], evidence_nodes, fusion_weight, intent)
            pred = result.answer
            evidence_text = " ".join(s.get("content", "") for s in (result.sources or []))
        except Exception as e:
            with print_lock:
                print(f"    [ERROR] sample {idx+1}: {e}")
            pred, evidence_text, fusion_weight = "", "", 0.5
            intent = IntentType.FACT_FINDING
        lat = time.perf_counter() - t0
        with print_lock:
            print(f"  [{idx+1}/{n}] {sample['question'][:70]}  ({lat:.1f}s)")
        return idx, sample, pred, evidence_text, fusion_weight, intent, lat

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, i, s): i for i, s in enumerate(samples)}
        for fut in as_completed(futures):
            idx, sample, pred, evidence_text, fusion_weight, intent, lat = fut.result()
            results[idx] = (sample, pred, evidence_text, fusion_weight, intent)
            latencies[idx] = lat

    predictions, ground_truths, log = [], [], []
    for (sample, pred, evidence_text, fusion_weight, intent), lat in zip(results, latencies):
        predictions.append({
            "answer":   pred,
            "evidence": evidence_text,
            "intent":   intent.name.lower() if hasattr(intent, "name") else str(intent),
        })
        ground_truths.append({
            "answer":             sample["answer"],
            "intent":             sample.get("intent", "fact"),
            "execution_required": sample.get("execution_required", False),
            "is_cross_doc":       sample.get("is_cross_doc", False),
            "is_cross_year":      sample.get("is_cross_year", False),
            "is_hybrid_modal":    sample.get("is_hybrid_modal", False),
            "subset":             sample.get("subset", ""),
            "difficulty":         sample.get("difficulty", ""),
        })
        log.append({
            "question":      sample["question"],
            "ground_truth":  sample["answer"],
            "prediction":    pred,
            "fusion_weight": fusion_weight,
            "intent":        intent.name.lower() if hasattr(intent, "name") else str(intent),
            "latency_s":     round(lat, 3),
        })

    metrics = evaluator.evaluate_dataset(predictions, ground_truths)
    if latencies:
        metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
    if torch.cuda.is_available():
        metrics["peak_gpu_memory_gb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    return log, metrics


def save_results(log, metrics, output_dir, variant, dataset, split):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pred_path = os.path.join(output_dir, f"ablation_{variant}_{dataset}_{split}_predictions_{ts}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    metrics_path = os.path.join(output_dir, f"ablation_{variant}_{dataset}_{split}_metrics_{ts}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    csv_path = os.path.join(output_dir, "ablation_results.csv")
    fieldnames = ["timestamp", "variant", "dataset", "split",
                  "f1", "em", "cross_doc_f1", "hybrid_modal_f1",
                  "avg_latency_s", "peak_gpu_memory_gb"]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = {"timestamp": ts, "variant": variant, "dataset": dataset, "split": split}
        row.update({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
        writer.writerow(row)

    print(f"  Predictions -> {pred_path}")
    print(f"  Metrics     -> {metrics_path}")
    print(f"  CSV         -> {csv_path}")


def print_metrics(metrics, variant):
    print(f"\n  {'='*50}")
    print(f"  Variant: {variant}")
    print(f"  {'='*50}")
    key_metrics = ["f1", "em", "cross_doc_f1", "hybrid_modal_f1", "avg_latency_s"]
    for k in key_metrics:
        if k in metrics:
            v = metrics[k]
            print(f"    {k:<25} {v:.4f}" if isinstance(v, float) else f"    {k:<25} {v}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

VARIANTS = [
    "full",
    "no_hierarchy",
    "no_alignment",
    "no_tapas",
    "no_tapex",
    "no_fusion",
    "no_l1_edges",
    "no_l2_section",
    "no_l3_table",
]

VARIANT_LABELS = {
    "full":          "HC-RAG (Full)",
    "no_hierarchy":  "w/o Three-Level Index",
    "no_alignment":  "w/o Cross-Modal Alignment",
    "no_tapas":      "w/o TAPAS",
    "no_tapex":      "w/o TAPAS",
    "no_fusion":     "w/o Query-Aware Fusion",
    "no_l1_edges":   "w/o L1 Cross-Doc Edges",
    "no_l2_section": "w/o L2 Section Nodes",
    "no_l3_table":   "w/o L3 Table Structure",
}


def main():
    parser = argparse.ArgumentParser(description="HC-RAG ablation study (Table 4)")
    parser.add_argument("--variant", choices=VARIANTS, default=None,
                        help="Single variant to run (default: all)")
    parser.add_argument("--dataset", default="multidoc2025",
                        help="Dataset to evaluate on (default: multidoc2025)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default="./data/benchmarks")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--workers", type=int, default=16,
                        help="Concurrent API threads (default: 16)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = BenchmarkEvaluator()

    try:
        samples = load_dataset_split(args.dataset, args.split, args.data_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    variants_to_run = [args.variant] if args.variant else VARIANTS

    summary = {}
    for variant in variants_to_run:
        label = VARIANT_LABELS[variant]
        print(f"\n{'='*60}")
        print(f"Ablation variant: {label}")
        print(f"{'='*60}")

        retriever, ctx_builder, generator = build_variant(variant, config, device)

        log, metrics = evaluate_variant(
            variant, retriever, ctx_builder, generator,
            evaluator, samples, max_samples=args.max_samples,
            workers=args.workers,
        )
        print_metrics(metrics, label)
        save_results(log, metrics, args.output_dir, variant, args.dataset, args.split)
        summary[label] = metrics

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Variant':<35} {'F1':>7} {'EM':>7} {'CrossDoc':>10} {'HybridM':>9}")
    print(f"{'-'*70}")
    for label, m in summary.items():
        print(f"{label:<35} "
              f"{m.get('f1', 0):>6.1f}% "
              f"{m.get('em', 0):>6.1f}% "
              f"{m.get('cross_doc_f1', 0):>9.1f}% "
              f"{m.get('hybrid_modal_f1', 0):>8.1f}%")

    print("\nAblation study complete.")


if __name__ == "__main__":
    main()


