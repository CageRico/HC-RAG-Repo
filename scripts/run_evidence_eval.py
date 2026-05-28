"""
E.3 Evidence Retrieval Evaluation for HC-RAG.

Computes evidence-level metrics (not answer-level) to prove that HC-RAG's
advantage comes from better evidence organization, not a stronger generator.

Metrics computed per method:
  - Doc Hit@5 / @10       : gold document found in top-k retrieved chunks
  - Section Hit@5 / @10   : gold section found in top-k retrieved chunks
  - Evidence Recall@5 / @10: fraction of gold evidence units covered in top-k
  - Table Hit@5           : a comparable table evidence trace is retrieved
  - Cross-doc Recall      : for multi-doc questions, fraction of gold docs covered

Gold evidence is derived from the dataset's existing fields:
  - evidence_section  -> gold section name
  - company + year    -> gold doc ID  (e.g. "AAPL_2024")
  - companies         -> all gold doc IDs for cross-doc questions
  - is_hybrid_modal   -> whether a table chunk is required

Usage:
  python scripts/run_evidence_eval.py --baseline bm25 --dataset multidoc2025
  python scripts/run_evidence_eval.py --all_baselines --dataset multidoc2025
  python scripts/run_evidence_eval.py --all_baselines --dataset multidoc2025 --top_k 10
  python scripts/run_evidence_eval.py --hcrag --dataset multidoc2025
"""

import os
import sys
import json
import csv
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any

import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_datasets import load_dataset_split
from scripts.run_baselines import (
    BASELINES, _clean_sample, _infer_doc_id, _get_context,
    _chunk_text, _bm25_retrieve, _dense_retrieve, _get_text_encoder,
    _get_index_chunks_with_meta,
)
from src.run_metadata import build_run_metadata, save_run_metadata


# ---------------------------------------------------------------------------
# Evidence metric helpers
# ---------------------------------------------------------------------------

def _normalize_section(s: str) -> str:
    """Lowercase and strip punctuation for fuzzy section matching."""
    import re
    return re.sub(r'[^a-z0-9 ]', '', s.lower()).strip()


def doc_hit_at_k(retrieved: List[Dict], gold_docs: List[str], k: int) -> float:
    """1 if any gold doc_id exactly matches a doc_id in top-k retrieved chunks."""
    if not gold_docs:
        return float("nan")
    top = retrieved[:k]
    gold_set = {d.lower() for d in gold_docs if d}
    if not gold_set:
        return float("nan")
    return float(any(r.get("doc_id", "").lower() in gold_set for r in top
                     if r.get("doc_id", "")))


def section_hit_at_k(retrieved: List[Dict], gold_section: str, k: int) -> float:
    """1 if gold section text appears (fuzzy) in any top-k chunk's section field or content."""
    if not gold_section:
        return float("nan")  # not applicable when no gold section
    gold_norm = _normalize_section(gold_section)
    if not gold_norm:
        return float("nan")
    # Use all meaningful tokens (>=4 chars) for matching to avoid false positives from "item"/"the"
    gold_tokens = [t for t in gold_norm.split() if len(t) >= 4]
    if not gold_tokens:
        gold_tokens = gold_norm.split()
    top = retrieved[:k]
    for r in top:
        section_norm = _normalize_section(r.get("section", ""))
        content_norm = _normalize_section(r.get("content", "")[:800])
        # Require majority of gold tokens to match
        matches = sum(1 for t in gold_tokens
                      if t in section_norm or t in content_norm)
        if matches >= max(1, len(gold_tokens) // 2):
            return 1.0
    return 0.0


def evidence_recall_at_k(retrieved: List[Dict], gold_section: str,
                          gold_docs: List[str], k: int) -> float:
    """
    Fraction of gold (doc, section) evidence units found in top-k.
    A gold unit is hit if:
      - its doc_id is found in top-k retrieved chunks, AND
      - if gold_section is provided, the section also matches (majority-token fuzzy match)
    """
    if not gold_docs:
        return float("nan")

    gold_sec_norm = _normalize_section(gold_section) if gold_section else ""
    gold_sec_tokens = [t for t in gold_sec_norm.split() if len(t) >= 4] if gold_sec_norm else []
    if not gold_sec_tokens and gold_sec_norm:
        gold_sec_tokens = gold_sec_norm.split()

    top = retrieved[:k]
    hits = 0
    for doc in gold_docs:
        doc_lower = doc.lower()
        # Find chunks from this doc
        doc_chunks = [r for r in top if r.get("doc_id", "").lower() == doc_lower]
        if not doc_chunks:
            continue
        if gold_sec_tokens:
            # Require section match in at least one chunk from this doc
            sec_hit = False
            for r in doc_chunks:
                section_norm = _normalize_section(r.get("section", ""))
                content_norm = _normalize_section(r.get("content", "")[:800])
                matches = sum(1 for t in gold_sec_tokens
                              if t in section_norm or t in content_norm)
                if matches >= max(1, len(gold_sec_tokens) // 2):
                    sec_hit = True
                    break
            hits += 1 if sec_hit else 0
        else:
            hits += 1
    return hits / len(gold_docs)


def table_hit_at_k(retrieved: List[Dict], is_hybrid: bool, k: int) -> float:
    """
    For hybrid-modal questions, 1 if any top-k chunk exposes a comparable
    table evidence trace. The released schema accepts explicit table tags,
    table_id metadata, or reconstructed table-like spans.
    """
    import re as _re
    if not is_hybrid:
        return float("nan")  # not applicable
    top = retrieved[:k]
    for r in top:
        content = r.get("content", "")
        chunk_type = r.get("type", "")
        # Explicit table type tag
        if chunk_type in ("table", "table_cell", "table_row"):
            return 1.0
        # Heuristic: multiple pipe-separated columns OR dense numeric rows
        lines = content[:600].split("\n")
        table_lines = [l for l in lines if l.count("|") >= 2
                       or len(_re.findall(r'\b\d[\d,\.]+\b', l)) >= 3]
        if len(table_lines) >= 2:
            return 1.0
    return 0.0


def cross_doc_recall(retrieved: List[Dict], gold_docs: List[str]) -> float:
    """
    For multi-doc questions: fraction of gold docs covered anywhere in retrieved.
    Returns nan for single-doc questions.
    """
    if len(gold_docs) <= 1:
        return float("nan")
    retrieved_docs = {r.get("doc_id", "").lower() for r in retrieved}
    hit = sum(1 for d in gold_docs if d.lower() in retrieved_docs)
    return hit / len(gold_docs)


# ---------------------------------------------------------------------------
# Gold evidence extraction from sample metadata
# ---------------------------------------------------------------------------

def _gold_from_sample(sample: dict):
    """
    Returns (gold_docs, gold_section, is_hybrid, is_cross_doc).
    gold_docs: list of doc IDs like ["AAPL_2024", "MSFT_2024"]
    """
    companies = sample.get("companies", [])
    years     = sample.get("years_required", [])
    year      = sample.get("year", "")

    # Build gold doc IDs
    if companies and len(companies) > 1:
        # Cross-doc: pair each company with the relevant year(s)
        if len(years) == 1:
            gold_docs = [f"{c}_{years[0]}" for c in companies]
        elif len(years) == len(companies):
            gold_docs = [f"{c}_{y}" for c, y in zip(companies, years)]
        else:
            gold_docs = [f"{c}_{year or (years[0] if years else '')}" for c in companies]
    elif companies:
        if years:
            gold_docs = [f"{companies[0]}_{y}" for y in years]
        else:
            gold_docs = [f"{companies[0]}_{year}"] if year else [companies[0]]
    else:
        doc_id = _infer_doc_id(sample)
        gold_docs = [doc_id] if doc_id != "unknown" else []

    gold_section = sample.get("evidence_section", "")
    is_hybrid    = sample.get("is_hybrid_modal", False)
    is_cross_doc = sample.get("is_cross_doc", False) or len(gold_docs) > 1

    return gold_docs, gold_section, is_hybrid, is_cross_doc


# ---------------------------------------------------------------------------
# HC-RAG retrieval wrapper (for E.3 comparison)
# ---------------------------------------------------------------------------

class HCRAGRetriever:
    """Wraps HCRAGInference to expose a retrieve() interface for E.3."""
    name = "hcrag"

    def __init__(self, config_path: str):
        from scripts.run_inference import HCRAGInference
        self.hcrag = HCRAGInference(config_path)
        self._idx = self.hcrag.index

    def _resolve_node(self, node) -> tuple:
        """Trace a node upward through reverse_edges and return (doc_id, section_title)."""
        idx = self._idx
        nid = node.node_id
        parents1 = idx.reverse_edges.get(nid, [])
        if not parents1:
            return "", ""
        sec_id = parents1[0][0]
        parents2 = idx.reverse_edges.get(sec_id, [])
        doc_id = parents2[0][0] if parents2 else ""
        sec_node = idx.section_nodes.get(sec_id) or idx.nodes.get(sec_id)
        if sec_node and getattr(sec_node, "metadata", None):
            sec_title = sec_node.metadata.get("title", sec_id)
        else:
            sec_title = sec_id
        return doc_id, sec_title

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        question = sample.get("question", "")
        inline_ctx = sample.get("context", "")
        if isinstance(inline_ctx, list):
            inline_ctx = "\n".join(str(x) for x in inline_ctx)
        inline_ctx = inline_ctx.strip()

        try:
            if inline_ctx:
                # Datasets with inline context, such as FinQA, use the context path.
                result = self.hcrag.answer_with_context(question, inline_ctx)
                sources = result.get("sources", [])[:top_k]
                doc_id = _infer_doc_id(sample)
                return [
                    {
                        "content": s.get("content", ""),
                        "doc_id":  s.get("doc_id", doc_id),
                        "section": s.get("section", s.get("title", "")),
                        "rank":    i + 1,
                        "type":    s.get("type", "text"),
                    }
                    for i, s in enumerate(sources)
                ]
            else:
                # Multi-Doc-2025 calls the retriever directly and resolves doc/section via reverse_edges.
                # Add company/year metadata to the query to improve L1 metadata matching.
                companies = sample.get("companies", [])
                years = sample.get("years_required", []) or ([str(sample["year"])] if sample.get("year") else [])
                ctx_prefix = ""
                if companies:
                    ctx_prefix += " ".join(companies) + " "
                if years:
                    ctx_prefix += " ".join(str(y) for y in years) + " "
                augmented_query = (ctx_prefix + question).strip()

                nodes, _, _ = self.hcrag.retriever.retrieve(augmented_query)
                results = []
                for i, node in enumerate(nodes[:top_k]):
                    doc_id, sec_title = self._resolve_node(node)
                    content = node.content
                    if isinstance(content, dict):
                        content = str(content)
                    results.append({
                        "content": content or "",
                        "doc_id":  doc_id,
                        "section": sec_title,
                        "rank":    i + 1,
                        "type":    node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type),
                    })
                return results
        except Exception as e:
            print(f"    [HC-RAG retrieve ERROR] {e}")
            return []


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_evidence(retriever, samples: List[Dict], top_k: int = 10,
                      max_samples: int = None, workers: int = 8,
                      checkpoint_path: str = None) -> Dict[str, Any]:
    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    results = [None] * n
    print_lock = threading.Lock()
    ckpt_lock = threading.Lock()

    # Resume support: load completed samples.
    completed_indices = set()
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    idx = entry["idx"]
                    results[idx] = entry["metrics"]
                    completed_indices.add(idx)
                except Exception:
                    pass
        if completed_indices:
            print(f"  [checkpoint] restored {len(completed_indices)}/{n} samples; continuing...")

    def _process(idx, raw_sample):
        sample = _clean_sample(raw_sample)
        # Restore metadata fields stripped by _clean_sample
        for field in ("companies", "years_required", "year", "company",
                      "evidence_section", "is_hybrid_modal", "is_cross_doc",
                      "is_cross_year", "subset", "difficulty", "sector", "intent"):
            if field in raw_sample and field not in sample:
                sample[field] = raw_sample[field]

        gold_docs, gold_section, is_hybrid, is_cross_doc = _gold_from_sample(sample)

        try:
            retrieved = retriever.retrieve(sample, top_k=top_k)
        except Exception as e:
            with print_lock:
                print(f"    [ERROR] sample {idx+1}: {e}")
            retrieved = []

        metrics = {
            "doc_hit_5":       doc_hit_at_k(retrieved, gold_docs, 5),
            "doc_hit_10":      doc_hit_at_k(retrieved, gold_docs, 10),
            "section_hit_5":   section_hit_at_k(retrieved, gold_section, 5),
            "section_hit_10":  section_hit_at_k(retrieved, gold_section, 10),
            "recall_5":        evidence_recall_at_k(retrieved, gold_section, gold_docs, 5),
            "recall_10":       evidence_recall_at_k(retrieved, gold_section, gold_docs, 10),
            "table_hit_5":     table_hit_at_k(retrieved, is_hybrid, 5),
            "cross_doc_recall": cross_doc_recall(retrieved, gold_docs),
            # metadata for slicing
            "is_cross_doc":    is_cross_doc,
            "is_cross_year":   sample.get("is_cross_year", False),
            "is_hybrid":       is_hybrid,
            "subset":          sample.get("subset", ""),
            "difficulty":      sample.get("difficulty", ""),
            "intent":          sample.get("intent", ""),
        }

        # Write each completed sample immediately to the checkpoint.
        if checkpoint_path:
            with ckpt_lock:
                with open(checkpoint_path, "a", encoding="utf-8") as cf:
                    cf.write(json.dumps({"idx": idx, "metrics": metrics}, ensure_ascii=False) + "\n")
                    cf.flush()
                    os.fsync(cf.fileno())

        def _fmt(v):
            import math
            return "nan" if (isinstance(v, float) and math.isnan(v)) else f"{v:.2f}"

        with print_lock:
            print(f"  [{idx+1}/{n}] doc_hit@5={_fmt(metrics['doc_hit_5'])} "
                  f"sec_hit@5={_fmt(metrics['section_hit_5'])} "
                  f"recall@5={_fmt(metrics['recall_5'])}  "
                  f"{sample['question'][:60]}")
        return idx, metrics

    pending = [(i, s) for i, s in enumerate(samples) if i not in completed_indices]
    if workers == 1:
        # Serial mode reduces memory and thermal pressure.
        for i, s in pending:
            idx, m = _process(i, s)
            results[idx] = m
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_process, i, s): i for i, s in pending}
            for fut in as_completed(futures):
                idx, m = fut.result()
                results[idx] = m

    return _aggregate(results)


def _mean_valid(vals: list) -> float:
    """Mean ignoring NaN values."""
    import math
    valid = [v for v in vals if not (isinstance(v, float) and math.isnan(v))]
    return sum(valid) / len(valid) * 100 if valid else 0.0


def _aggregate(results: List[Dict]) -> Dict[str, float]:
    import math

    scalar_keys = ["doc_hit_5", "doc_hit_10", "section_hit_5", "section_hit_10",
                   "recall_5", "recall_10", "table_hit_5", "cross_doc_recall"]

    agg = {k: _mean_valid([r[k] for r in results]) for k in scalar_keys}

    # Sliced metrics use consistent naming: {flag}_{metric}_{k}
    for flag, label in [("is_cross_doc", "cross_doc"), ("is_cross_year", "cross_year"),
                        ("is_hybrid", "hybrid")]:
        sub = [r for r in results if r.get(flag)]
        if sub:
            agg[f"{label}_recall_5"]      = _mean_valid([r["recall_5"]      for r in sub])
            agg[f"{label}_recall_10"]     = _mean_valid([r["recall_10"]     for r in sub])
            agg[f"{label}_section_hit_5"] = _mean_valid([r["section_hit_5"] for r in sub])
            agg[f"{label}_doc_hit_5"]     = _mean_valid([r["doc_hit_5"]     for r in sub])

    for subset in ["S1", "S2", "S3", "S4", "S5"]:
        sub = [r for r in results if r.get("subset") == subset]
        if sub:
            agg[f"subset_{subset}_recall_5"] = _mean_valid([r["recall_5"] for r in sub])

    for diff in ["L1", "L2", "L3", "L4"]:
        sub = [r for r in results if r.get("difficulty") == diff]
        if sub:
            agg[f"diff_{diff}_recall_5"] = _mean_valid([r["recall_5"] for r in sub])

    for intent in ["calculation", "trend", "fact", "comparison"]:
        sub = [r for r in results if r.get("intent") == intent]
        if sub:
            agg[f"intent_{intent}_recall_5"] = _mean_valid([r["recall_5"] for r in sub])

    agg["n_samples"] = len(results)
    return agg


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_evidence_results(metrics: Dict, output_dir: str, method: str,
                          dataset: str, split: str, top_k: int,
                          config: Dict[str, Any], config_path: str,
                          max_samples: int, workers: int):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    json_path = os.path.join(output_dir, f"evidence_{method}_{dataset}_{split}_{ts}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"method": method, "dataset": dataset, "split": split,
                   "top_k": top_k, "metrics": metrics}, f, ensure_ascii=False, indent=2)

    metadata = build_run_metadata(
        config=config,
        config_path=config_path,
        script_name="scripts/run_evidence_eval.py",
        run_type="e3_evidence_eval",
        dataset=dataset,
        split=split,
        method=method,
        output_dir=output_dir,
        retrieval_top_k=top_k,
        final_evidence_budget=top_k,
        max_samples=max_samples,
        workers=workers,
        extra={
            "n_samples_evaluated": metrics.get("n_samples"),
            "reported_cutoffs": [5, 10],
            "evidence_localization_mode": "fixed_reported_cutoffs",
        },
    )
    metadata_path = save_run_metadata(metadata, output_dir, f"evidence_{method}_{dataset}_{split}", ts)

    # Master CSV for easy table construction
    csv_path = os.path.join(output_dir, "evidence_results.csv")
    fieldnames = [
        "timestamp", "method", "dataset", "split", "top_k", "n_samples",
        # E3 main table columns (Table 4)
        "doc_hit_5", "doc_hit_10",
        "section_hit_5", "section_hit_10",
        "recall_5", "recall_10",
        "table_hit_5",
        "cross_doc_recall",
        # sliced by evidence structure
        "cross_doc_recall_5", "cross_doc_recall_10",
        "cross_doc_section_hit_5", "cross_doc_doc_hit_5",
        "cross_year_recall_5", "cross_year_recall_10",
        "hybrid_recall_5", "hybrid_recall_10",
        # subset slices (Multi-Doc-2025)
        "subset_S1_recall_5", "subset_S2_recall_5", "subset_S3_recall_5",
        "subset_S4_recall_5", "subset_S5_recall_5",
        # intent slices
        "intent_calculation_recall_5", "intent_trend_recall_5",
        "intent_fact_recall_5", "intent_comparison_recall_5",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = {"timestamp": ts, "method": method, "dataset": dataset,
               "split": split, "top_k": top_k}
        row.update({k: f"{v:.2f}" if isinstance(v, float) else v
                    for k, v in metrics.items()})
        writer.writerow(row)

    print(f"  JSON    -> {json_path}")
    print(f"  Run Meta-> {metadata_path}")
    print(f"  CSV     -> {csv_path}")


def print_evidence_metrics(metrics: Dict, method: str):
    print(f"\n{'='*55}")
    print(f"Evidence Retrieval Results: {method}")
    print(f"{'='*55}")
    # E3 main table columns (matches Table 4)
    main_keys = [
        "doc_hit_5", "section_hit_5", "recall_5", "recall_10",
        "table_hit_5", "cross_doc_recall",
    ]
    print("  [Main Metrics]")
    for k in main_keys:
        if k in metrics:
            print(f"  {k:<30} {metrics[k]:.2f}%")
    # Sliced metrics
    slice_keys = [k for k in sorted(metrics.keys())
                  if k not in main_keys and k != "n_samples"
                  and any(k.startswith(p) for p in
                          ("cross_doc_", "cross_year_", "hybrid_", "subset_",
                           "diff_", "intent_"))]
    if slice_keys:
        print("  [Sliced Metrics]")
        for k in slice_keys:
            v = metrics[k]
            print(f"  {k:<35} {v:.2f}%" if isinstance(v, float) else f"  {k:<35} {v}")
    print(f"  n_samples: {metrics.get('n_samples', '?')}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DATASETS = ["multidoc2025", "finqa", "tatqa", "financebench", "docfinqa"]


def main():
    parser = argparse.ArgumentParser(description="E.3 Evidence Retrieval Evaluation")
    parser.add_argument("--baseline", choices=list(BASELINES.keys()), default="bm25")
    parser.add_argument("--all_baselines", action="store_true",
                        help="Run all baseline retrievers")
    parser.add_argument("--all_methods", action="store_true",
                        help="Run all baselines AND HC-RAG (equivalent to --all_baselines --hcrag)")
    parser.add_argument("--hcrag", action="store_true", help="Also evaluate HC-RAG retrieval")
    parser.add_argument("--dataset", choices=DATASETS, default="multidoc2025")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default="./data/benchmarks")
    parser.add_argument("--output_dir", default="./outputs/evidence_eval")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--resume", action="store_true",
                        help="Resume HC-RAG evidence evaluation from checkpoint")
    args = parser.parse_args()

    # --all_methods implies both --all_baselines and --hcrag
    if args.all_methods:
        args.all_baselines = True
        args.hcrag = True

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loading dataset: {args.dataset}/{args.split} ...")
    try:
        samples = load_dataset_split(args.dataset, args.split, args.data_dir)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return

    methods_to_run = list(BASELINES.keys()) if args.all_baselines else [args.baseline]

    for bl_name in methods_to_run:
        print(f"\n{'='*55}")
        print(f"Method: {bl_name}")
        print(f"{'='*55}")
        retriever = BASELINES[bl_name](config)
        metrics = evaluate_evidence(
            retriever, samples,
            top_k=args.top_k,
            max_samples=args.max_samples,
            workers=args.workers,
        )
        print_evidence_metrics(metrics, bl_name)
        save_evidence_results(
            metrics, args.output_dir, bl_name, args.dataset, args.split, args.top_k,
            config, args.config, args.max_samples, args.workers,
        )

    if args.hcrag:
        print(f"\n{'='*55}")
        print("Method: hcrag")
        print(f"{'='*55}")
        # Use a fixed checkpoint filename for resume support.
        ckpt_path = os.path.join(args.output_dir, f"hcrag_{args.dataset}_{args.split}_checkpoint.jsonl")
        if not args.resume and os.path.exists(ckpt_path):
            # Clear stale checkpoints when not running in resume mode.
            os.remove(ckpt_path)
            print("  [checkpoint] cleared stale checkpoint; starting fresh")
        try:
            retriever = HCRAGRetriever(args.config)
            metrics = evaluate_evidence(
                retriever, samples,
                top_k=args.top_k,
                max_samples=args.max_samples,
                workers=args.workers,
                checkpoint_path=ckpt_path,
            )
            print_evidence_metrics(metrics, "hcrag")
            save_evidence_results(
                metrics, args.output_dir, "hcrag", args.dataset, args.split, args.top_k,
                config, args.config, args.max_samples, args.workers,
            )
            # Delete checkpoint after successful completion.
            if os.path.exists(ckpt_path):
                os.remove(ckpt_path)
                print("  [checkpoint] completed; checkpoint removed")
        except Exception as e:
            print(f"  [HC-RAG ERROR] {e}")
            print(f"  [checkpoint] progress saved to {ckpt_path}; use --resume to continue")

    print("\nEvidence evaluation complete.")
    print(f"Results saved to: {args.output_dir}/evidence_results.csv")


if __name__ == "__main__":
    main()
