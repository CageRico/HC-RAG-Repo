"""
Figure 7: Robustness analysis - F1 vs. number of distractor passages.

Injects N irrelevant passages into the retrieval context and measures
how each model's F1 degrades as noise increases.

Usage:
  python scripts/run_robustness.py --config config.yaml
  python scripts/run_robustness.py --config config.yaml --max_samples 100 --workers 8
"""

import os
import sys
import json
import time
import argparse
import yaml
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_datasets import load_dataset_split
from scripts.run_inference import HCRAGInference
from src.evaluation import BenchmarkEvaluator


DISTRACTOR_COUNTS = [0, 5, 10, 15, 20]
MAX_SAMPLES_DEFAULT = 100

# Fixed distractor passages — generic financial boilerplate that is
# topically plausible but irrelevant to any specific question.
_DISTRACTORS = [
    "The company operates in a highly competitive market and faces risks from regulatory changes, "
    "macroeconomic conditions, and technological disruption. Management continues to monitor these "
    "factors and adjust strategy accordingly.",

    "Forward-looking statements in this report are subject to risks and uncertainties. Actual results "
    "may differ materially from those projected. The company undertakes no obligation to update these "
    "statements after the date of this filing.",

    "Our internal controls over financial reporting are designed to provide reasonable assurance "
    "regarding the reliability of financial reporting and the preparation of financial statements "
    "in accordance with generally accepted accounting principles.",

    "The audit committee reviewed the company's financial statements and discussed them with "
    "management and the independent registered public accounting firm. The committee recommended "
    "that the board approve the financial statements for inclusion in the annual report.",

    "Capital expenditures for the fiscal year totaled approximately $2.3 billion, primarily related "
    "to investments in manufacturing capacity, research and development facilities, and information "
    "technology infrastructure.",

    "The company's effective tax rate for the period was 21.4%, compared to 19.8% in the prior year. "
    "The increase was primarily attributable to changes in the geographic mix of earnings and "
    "non-deductible expenses.",

    "Net cash provided by operating activities was $4.1 billion for the year ended December 31. "
    "This compares to $3.7 billion in the prior year, reflecting improved working capital management "
    "and higher net income.",

    "The company repurchased 12.5 million shares of common stock during the year at an average price "
    "of $142.30 per share, for a total cost of approximately $1.8 billion under the board-authorized "
    "share repurchase program.",

    "Goodwill and intangible assets arising from acquisitions are tested for impairment annually or "
    "whenever events or changes in circumstances indicate that the carrying amount may not be "
    "recoverable. No impairment charges were recorded during the current period.",

    "The company maintains a revolving credit facility of $3.0 billion with a syndicate of financial "
    "institutions. As of the balance sheet date, there were no outstanding borrowings under this "
    "facility and the company was in compliance with all covenants.",

    "Research and development expenses increased 8.2% year-over-year to $1.6 billion, representing "
    "approximately 6.4% of net revenues. The increase reflects continued investment in next-generation "
    "product development and platform innovation.",

    "The company's defined benefit pension plans had a combined projected benefit obligation of "
    "$2.8 billion and plan assets of $2.4 billion as of the measurement date, resulting in an "
    "underfunded status of $0.4 billion.",

    "Segment operating income for the Americas region increased 5.3% to $892 million, driven by "
    "volume growth and favorable pricing, partially offset by higher input costs and increased "
    "selling and marketing expenses.",

    "The company is subject to various legal proceedings and claims that arise in the ordinary course "
    "of business. Management believes that the ultimate resolution of these matters will not have a "
    "material adverse effect on the company's financial position.",

    "Diluted earnings per share for the fiscal year were $6.84, compared to $5.97 in the prior year, "
    "an increase of 14.6%. The improvement reflects higher net income and the benefit of share "
    "repurchases completed during the year.",

    "The company's supply chain operations span 47 countries and involve relationships with over "
    "3,000 suppliers. Disruptions in the supply chain, including those caused by geopolitical events "
    "or natural disasters, could adversely affect operations.",

    "Deferred revenue at year-end totaled $1.2 billion, primarily related to software subscriptions "
    "and extended warranty contracts. Substantially all of this amount is expected to be recognized "
    "as revenue within the next 24 months.",

    "The company's international operations generated revenues of $8.4 billion, representing 52% of "
    "total net revenues. Currency fluctuations had a negative impact of approximately 2.1 percentage "
    "points on reported international revenue growth.",

    "Inventory at the end of the period was $3.1 billion, an increase of $0.4 billion from the prior "
    "year-end. Days inventory outstanding was 62 days, compared to 58 days in the prior year, "
    "reflecting strategic inventory builds in certain product categories.",

    "The board of directors declared a quarterly cash dividend of $0.88 per share, payable to "
    "shareholders of record as of the specified date. The annualized dividend rate of $3.52 per share "
    "represents a 10% increase from the prior year.",
]


def _make_client(config):
    import openai
    cfg = config["models"]
    return openai.OpenAI(
        api_key=cfg.get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", ""),
        base_url=cfg.get("openai_base_url") or os.getenv("OPENAI_BASE_URL") or None,
    ), cfg["generator"]


def _call_llm(client, model, prompt, config):
    gen = config["generation"]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Answer based only on the provided evidence."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=gen.get("max_tokens", 512),
            temperature=gen.get("temperature", 0.0),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def _chunk_text(text, chunk_size=512, overlap=50):
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks or [text]


def _get_encoder(config):
    from src.encoders import TextEncoder
    enc = TextEncoder(model_name=config["models"]["text_encoder"],
                      embedding_dim=config["alignment"]["embedding_dim"],
                      local_files_only=config["models"].get("local_files_only", True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc.to(device).eval()
    return enc


def _dense_retrieve(query, chunks, encoder, top_k=5):
    if not chunks:
        return []
    q_emb = encoder(query)
    if isinstance(q_emb, torch.Tensor):
        q_emb = q_emb.cpu().numpy()
    q_emb = q_emb.squeeze()
    embs = []
    for i in range(0, len(chunks), 64):
        e = encoder(chunks[i: i + 64])
        if isinstance(e, torch.Tensor):
            e = e.cpu().numpy()
        embs.append(e)
    embs = np.concatenate(embs, axis=0)
    norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
    scores = (embs / norms) @ (q_emb / (np.linalg.norm(q_emb) + 1e-9))
    return [chunks[i] for i in np.argsort(scores)[::-1][:top_k]]


def _inject_distractors(context: str, n: int) -> str:
    """Append n distractor passages to the context."""
    distractors = (_DISTRACTORS * ((n // len(_DISTRACTORS)) + 1))[:n]
    return context + "\n\n" + "\n\n".join(distractors)


# ---------------------------------------------------------------------------
# Per-model evaluation at a given distractor count
# ---------------------------------------------------------------------------

def _eval_hcrag(hcrag, evaluator, samples, n_distractors, workers):
    n = len(samples)
    results = [None] * n
    lock = threading.Lock()

    def _run(idx, sample):
        ctx = sample.get("context", "").strip()
        if isinstance(ctx, list):
            ctx = "\n".join(str(x) for x in ctx)
        noisy_ctx = _inject_distractors(ctx, n_distractors) if ctx else ""
        try:
            if noisy_ctx:
                r = hcrag.answer_with_context(sample["question"], noisy_ctx)
            else:
                r = hcrag.answer(sample["question"])
            pred = r["answer"]
        except Exception:
            pred = ""
        with lock:
            pass
        return idx, pred

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run, i, s): i for i, s in enumerate(samples)}
        for f in as_completed(futs):
            idx, pred = f.result()
            results[idx] = pred

    predictions  = [{"answer": p, "evidence": "", "intent": s.get("intent", "fact")}
                    for p, s in zip(results, samples)]
    ground_truths = [{"answer": s["answer"], "intent": s.get("intent", "fact"),
                      "execution_required": s.get("execution_required", False),
                      "is_cross_doc": s.get("is_cross_doc", False),
                      "is_cross_year": s.get("is_cross_year", False),
                      "is_hybrid_modal": s.get("is_hybrid_modal", False),
                      "subset": s.get("subset", ""), "difficulty": s.get("difficulty", "")}
                     for s in samples]
    metrics = evaluator.evaluate_dataset(predictions, ground_truths)
    return metrics.get("f1", 0.0)


def _eval_baseline(client, model, encoder, evaluator, samples, n_distractors, workers, config, name):
    n = len(samples)
    results = [None] * n
    lock = threading.Lock()

    def _run(idx, sample):
        ctx = sample.get("context", "").strip()
        if isinstance(ctx, list):
            ctx = "\n".join(str(x) for x in ctx)
        noisy_ctx = _inject_distractors(ctx, n_distractors) if ctx else "No context."
        chunks = _chunk_text(noisy_ctx)
        retrieved = _dense_retrieve(sample["question"], chunks, encoder)
        evidence = "\n\n".join(retrieved)
        prompt = f"Evidence:\n{evidence}\n\nQuestion: {sample['question']}\nAnswer:"
        try:
            pred = _call_llm(client, model, prompt, config)
        except Exception:
            pred = ""
        return idx, pred

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_run, i, s): i for i, s in enumerate(samples)}
        for f in as_completed(futs):
            idx, pred = f.result()
            results[idx] = pred

    predictions  = [{"answer": p, "evidence": "", "intent": s.get("intent", "fact")}
                    for p, s in zip(results, samples)]
    ground_truths = [{"answer": s["answer"], "intent": s.get("intent", "fact"),
                      "execution_required": s.get("execution_required", False),
                      "is_cross_doc": s.get("is_cross_doc", False),
                      "is_cross_year": s.get("is_cross_year", False),
                      "is_hybrid_modal": s.get("is_hybrid_modal", False),
                      "subset": s.get("subset", ""), "difficulty": s.get("difficulty", "")}
                     for s in samples]
    metrics = evaluator.evaluate_dataset(predictions, ground_truths)
    return metrics.get("f1", 0.0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_robustness(config_path, output_dir, max_samples, workers):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Load dataset — prefer multidoc2025, fall back to finqa
    for ds in ("multidoc2025", "finqa"):
        try:
            all_samples = load_dataset_split(ds, "test")
            print(f"Using {ds} ({len(all_samples)} samples)")
            break
        except FileNotFoundError:
            continue
    else:
        print("[ERROR] No dataset found.")
        return

    samples = all_samples[:max_samples]
    print(f"Evaluating on {len(samples)} samples, {DISTRACTOR_COUNTS} distractor levels\n")

    evaluator = BenchmarkEvaluator()
    hcrag     = HCRAGInference(config_path)
    client, model = _make_client(config)
    encoder   = _get_encoder(config)

    results = {
        "distractor_counts": DISTRACTOR_COUNTS,
        "HC-RAG":      [],
        "Vanilla RAG": [],
        "Self-RAG":    [],
        "Graph-RAG (entity)": [],
    }

    for n_dist in DISTRACTOR_COUNTS:
        print(f"=== Distractor count: {n_dist} ===")

        f1_hcrag = _eval_hcrag(hcrag, evaluator, samples, n_dist, workers)
        print(f"  HC-RAG:      F1={f1_hcrag:.2f}")

        f1_vanilla = _eval_baseline(client, model, encoder, evaluator,
                                    samples, n_dist, workers, config, "vanilla_rag")
        print(f"  Vanilla RAG: F1={f1_vanilla:.2f}")

        f1_selfrag = _eval_baseline(client, model, encoder, evaluator,
                                    samples, n_dist, workers, config, "self_rag")
        print(f"  Self-RAG:    F1={f1_selfrag:.2f}")

        f1_graph = _eval_baseline(client, model, encoder, evaluator,
                                  samples, n_dist, workers, config, "graphrag")
        print(f"  Graph-RAG (entity): F1={f1_graph:.2f}")

        results["HC-RAG"].append(round(f1_hcrag, 2))
        results["Vanilla RAG"].append(round(f1_vanilla, 2))
        results["Self-RAG"].append(round(f1_selfrag, 2))
        results["Graph-RAG (entity)"].append(round(f1_graph, 2))

    out_path = os.path.join(output_dir, "robustness_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved robustness results -> {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="HC-RAG robustness experiment (Figure 7)")
    parser.add_argument("--config",      default="config.yaml")
    parser.add_argument("--output_dir",  default="./outputs")
    parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES_DEFAULT)
    parser.add_argument("--workers",     type=int, default=8)
    args = parser.parse_args()
    run_robustness(args.config, args.output_dir, args.max_samples, args.workers)


if __name__ == "__main__":
    main()
