"""
End-to-end evaluation of HC-RAG on benchmark datasets.

Steps:
  1. Load converted dataset (from prepare_datasets.py)
  2. Run HC-RAG inference on each question
  3. Compute EM / F1 / Exec-Acc / Hallucination-Rate
  4. Save per-sample predictions and aggregate metrics to outputs/

Usage:
  python scripts/run_evaluation.py --dataset finqa --split test
  python scripts/run_evaluation.py --dataset tatqa --split test --max_samples 200
  python scripts/run_evaluation.py --all_datasets
"""

import os
import sys
import json
import argparse
import csv
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any

import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_datasets import load_dataset_split
from scripts.run_inference import HCRAGInference
from src.evaluation import BenchmarkEvaluator
from src.run_metadata import build_run_metadata, save_run_metadata


def _extract_question(raw: str) -> str:
    """
    flare-* datasets embed the question inside a prompt like:
      'Please answer ... Context: <ctx>\nQuestion: <q>\nAnswer:'
    Extract just the question text.
    """
    if not isinstance(raw, str):
        return str(raw)
    # Try to find explicit "Question:" marker
    for marker in ("Question:", "question:", "Q:"):
        idx = raw.rfind(marker)
        if idx != -1:
            tail = raw[idx + len(marker):].strip()
            # Strip trailing "Answer:" if present
            for end in ("\nAnswer:", "\nA:", "\n"):
                if end in tail:
                    tail = tail[:tail.index(end)].strip()
                    break
            if tail:
                return tail
    # Fallback: if the string contains "\nContext:" strip everything before the last newline
    if "\nContext:" in raw:
        # The question is usually the first line
        first_line = raw.split("\n")[0].strip()
        if len(first_line) > 10:
            return first_line
    return raw.strip()


def _clean_sample(sample: dict) -> dict:
    """Normalise question/answer fields to plain strings."""
    q = sample.get("question", "")
    a = sample.get("answer", "")
    sample = dict(sample)
    sample["question"] = _extract_question(q) if isinstance(q, str) else str(q)
    if isinstance(a, (list, dict)):
        sample["answer"] = ", ".join(str(x) for x in a) if isinstance(a, list) else str(a)
    else:
        sample["answer"] = str(a)
    return sample



def _gpu_memory_gb() -> float:
    """Return current peak GPU memory in GB, or 0 if no GPU."""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 3)
    return 0.0


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

def evaluate_dataset(
    hcrag: HCRAGInference,
    evaluator: BenchmarkEvaluator,
    samples: List[Dict],
    max_samples: int = None,
    workers: int = 16,
) -> tuple[List[Dict], Dict[str, float]]:
    """
    Run inference on every sample (concurrently) and compute metrics.
    workers: number of parallel API threads.
    """
    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    results = [None] * n
    latencies = [0.0] * n
    print_lock = threading.Lock()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _process(idx, raw_sample):
        sample = _clean_sample(raw_sample)
        question      = sample["question"]
        gt_answer     = sample["answer"]
        intent_gt     = sample.get("intent", "fact")
        exec_required = sample.get("execution_required", False)
        inline_context = sample.get("context", "")
        if isinstance(inline_context, list):
            inline_context = "\n".join(str(x) for x in inline_context)
        inline_context = inline_context.strip()

        t0 = time.perf_counter()
        try:
            # Datasets with self-contained context (FinQA, TAT-QA, FinanceBench, DocFinQA):
            # use context-based dense retrieval + cross-modal fusion.
            # Multi-Doc-2025 has no context field → use hierarchical index.
            if inline_context:
                result = hcrag.answer_with_context(question, inline_context)
            else:
                # Augment query with company/year metadata for index-based retrieval
                companies = raw_sample.get("companies", []) or ([raw_sample["company"]] if raw_sample.get("company") else [])
                years = raw_sample.get("years_required", []) or ([str(raw_sample["year"])] if raw_sample.get("year") else [])
                ctx_prefix = " ".join(companies) + " " + " ".join(str(y) for y in years)
                augmented_query = (ctx_prefix.strip() + " " + question).strip()
                result = hcrag.answer(augmented_query)
            pred_answer = result["answer"]
            evidence_text = " ".join(
                s.get("content", "") for s in result.get("sources", [])
            )
        except Exception as e:
            with print_lock:
                print(f"    [ERROR] sample {idx+1}: {e}")
            result, pred_answer, evidence_text = {}, "", ""
        lat = time.perf_counter() - t0

        with print_lock:
            print(f"  [{idx+1}/{n}] {question[:70]}  ({lat:.1f}s)")

        return idx, sample, question, gt_answer, intent_gt, exec_required, result, pred_answer, evidence_text, lat

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, i, s): i for i, s in enumerate(samples)}
        for fut in as_completed(futures):
            (idx, sample, question, gt_answer, intent_gt,
             exec_required, result, pred_answer, evidence_text, lat) = fut.result()
            results[idx] = (sample, question, gt_answer, intent_gt,
                            exec_required, result, pred_answer, evidence_text)
            latencies[idx] = lat

    predictions, ground_truths, predictions_log = [], [], []
    for (sample, question, gt_answer, intent_gt,
         exec_required, result, pred_answer, evidence_text), lat in zip(results, latencies):
        predictions.append({
            "answer":   pred_answer,
            "evidence": evidence_text,
            "intent":   result.get("intent", intent_gt),
        })
        ground_truths.append({
            "answer":             gt_answer,
            "intent":             intent_gt,
            "execution_required": exec_required,
            "is_cross_doc":       sample.get("is_cross_doc",    False),
            "is_cross_year":      sample.get("is_cross_year",   False),
            "is_hybrid_modal":    sample.get("is_hybrid_modal", False),
            "subset":             sample.get("subset",          ""),
            "difficulty":         sample.get("difficulty",      ""),
            "sector":             sample.get("sector",          ""),
            "companies":          sample.get("companies",       []),
        })
        predictions_log.append({
            "question":           question,
            "ground_truth":       gt_answer,
            "prediction":         pred_answer,
            "intent":             intent_gt,
            "execution_required": exec_required,
            "fusion_weight":      result.get("fusion_weight"),
            "confidence":         result.get("confidence"),
            "retrieval_top_k":    result.get("retrieval_top_k"),
            "final_evidence_budget": result.get("final_evidence_budget"),
            "retrieval_mode":     result.get("retrieval_mode"),
            "latency_s":          round(lat, 3),
        })

    metrics = evaluator.evaluate_dataset(predictions, ground_truths)

    if latencies:
        metrics["avg_latency_s"]    = round(sum(latencies) / len(latencies), 3)
        metrics["median_latency_s"] = round(sorted(latencies)[len(latencies) // 2], 3)
    metrics["peak_gpu_memory_gb"] = round(_gpu_memory_gb(), 2)

    return predictions_log, metrics


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_results(
    predictions_log: List[Dict],
    metrics: Dict[str, float],
    output_dir: str,
    dataset: str,
    split: str,
    config: Dict[str, Any],
    config_path: str,
    max_samples: int,
    workers: int,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Per-sample predictions JSON
    pred_path = os.path.join(output_dir, f"{dataset}_{split}_predictions_{timestamp}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions_log, f, ensure_ascii=False, indent=2)

    # Aggregate metrics JSON
    metrics_path = os.path.join(output_dir, f"{dataset}_{split}_metrics_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    retrieval_top_k_values = sorted({
        int(p["retrieval_top_k"]) for p in predictions_log
        if p.get("retrieval_top_k") is not None
    })
    final_budget_values = sorted({
        int(p["final_evidence_budget"]) for p in predictions_log
        if p.get("final_evidence_budget") is not None
    })
    retrieval_modes = sorted({
        str(p["retrieval_mode"]) for p in predictions_log
        if p.get("retrieval_mode")
    })
    metadata = build_run_metadata(
        config=config,
        config_path=config_path,
        script_name="scripts/run_evaluation.py",
        run_type="e2_answer_eval",
        dataset=dataset,
        split=split,
        method="hc_rag",
        output_dir=output_dir,
        retrieval_top_k=retrieval_top_k_values,
        final_evidence_budget=final_budget_values,
        max_samples=max_samples,
        workers=workers,
        extra={
            "n_samples_evaluated": len(predictions_log),
            "retrieval_modes_observed": retrieval_modes,
            "context_budget_words": 3000,
            "fairness_controls": {
                "shared_generator": True,
                "shared_prompt_template": True,
                "shared_decoding": True,
                "shared_max_context_length": True,
                "shared_evidence_serialization": True,
            },
        },
    )
    metadata_path = save_run_metadata(metadata, output_dir, f"{dataset}_{split}", timestamp)

    # Append one row to a master CSV for easy comparison across runs.
    # Use a fixed superset of columns so rows from different datasets align.
    csv_path = os.path.join(output_dir, "all_results.csv")
    all_metric_keys = [
        "em", "f1", "exec_acc", "hall_rate", "faithful_acc",
        # intent slices
        "calculation_em", "calculation_f1", "calculation_exec_acc",
        "trend_em", "trend_f1",
        "fact_em", "fact_f1",
        "comparison_em", "comparison_f1",
        # structural slices
        "cross_doc_f1", "cross_year_f1", "hybrid_modal_f1",
        "single_doc_f1", "cross_company_f1",
        # subset slices (Multi-Doc-2025)
        "subset_S1_f1", "subset_S2_f1", "subset_S3_f1", "subset_S4_f1", "subset_S5_f1",
        # difficulty slices
        "difficulty_L1_f1", "difficulty_L2_f1", "difficulty_L3_f1", "difficulty_L4_f1",
        # efficiency
        "avg_latency_s", "median_latency_s", "peak_gpu_memory_gb",
    ]
    fieldnames = ["timestamp", "dataset", "split"] + all_metric_keys
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = {"timestamp": timestamp, "dataset": dataset, "split": split}
        row.update({k: f"{v:.3f}" if isinstance(v, float) else v
                    for k, v in metrics.items()})
        writer.writerow(row)

    print(f"\n  Predictions -> {pred_path}")
    print(f"  Metrics     -> {metrics_path}")
    print(f"  Run Meta    -> {metadata_path}")
    print(f"  Master CSV  -> {csv_path}")


def print_metrics(metrics: Dict[str, float], dataset: str):
    print(f"\n{'='*50}")
    print(f"Results: {dataset}")
    print(f"{'='*50}")
    for k, v in sorted(metrics.items()):
        print(f"  {k:<25} {v:.2f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

DATASETS = ["finqa", "tatqa", "financebench", "docfinqa", "multidoc2025"]


def main():
    parser = argparse.ArgumentParser(description="Evaluate HC-RAG on benchmarks")
    parser.add_argument("--dataset", choices=DATASETS, default="finqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--all_datasets", action="store_true",
                        help="Run evaluation on all four datasets")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit number of samples (useful for quick testing)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default="./data/benchmarks")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--workers", type=int, default=16,
                        help="Concurrent API threads (default: 16)")
    args = parser.parse_args()

    print("Initializing HC-RAG ...")
    hcrag = HCRAGInference(args.config)
    evaluator = BenchmarkEvaluator()

    datasets_to_run = DATASETS if args.all_datasets else [args.dataset]

    for dataset in datasets_to_run:
        print(f"\nEvaluating {dataset} / {args.split} ...")
        try:
            samples = load_dataset_split(dataset, args.split, args.data_dir)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        predictions_log, metrics = evaluate_dataset(
            hcrag, evaluator, samples,
            max_samples=args.max_samples,
            workers=args.workers,
        )
        print_metrics(metrics, dataset)
        save_results(
            predictions_log, metrics, args.output_dir, dataset, args.split,
            hcrag.config, args.config, args.max_samples, args.workers,
        )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
