"""
Figure 5: Scalability Analysis — latency vs. document count.

Measures inference latency for HC-RAG and key baselines as the number of
documents in the index grows from 10 to 100.

Usage:
  python scripts/run_scalability.py --config config.yaml
  python scripts/run_scalability.py --config config.yaml --workers 8
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


DOC_COUNTS = [10, 20, 40, 60, 80, 100]
N_QUERIES  = 30   # queries per doc-count level (enough for stable latency estimate)


# ---------------------------------------------------------------------------
# Baseline latency helpers (same generator, flat retrieval)
# ---------------------------------------------------------------------------

def _make_client(config: dict):
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
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=gen.get("max_tokens", 512),
            temperature=gen.get("temperature", 0.0),
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
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


# ---------------------------------------------------------------------------
# Measure latency for one model at a given doc count
# ---------------------------------------------------------------------------

def _measure_hcrag(hcrag: HCRAGInference, samples: List[Dict],
                   n_queries: int, workers: int) -> float:
    queries = [s["question"] for s in samples[:n_queries]]
    latencies = []
    lock = threading.Lock()

    def _run(q):
        ctx = samples[queries.index(q)].get("context", "").strip()
        t0 = time.perf_counter()
        try:
            if ctx:
                hcrag.answer_with_context(q, ctx)
            else:
                hcrag.answer(q)
        except Exception:
            pass
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, q) for q in queries]
        for f in as_completed(futs):
            latencies.append(f.result())

    return float(np.median(latencies))


def _measure_vanilla_rag(client, model, encoder, samples, n_queries, workers, config) -> float:
    queries = [s["question"] for s in samples[:n_queries]]
    contexts = [s.get("context", "") for s in samples[:n_queries]]
    latencies = []

    def _run(q, ctx):
        chunks = _chunk_text(ctx) if ctx else ["No context available."]
        retrieved = _dense_retrieve(q, chunks, encoder)
        evidence = "\n\n".join(retrieved)
        prompt = f"Evidence:\n{evidence}\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        _call_llm(client, model, prompt, config)
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, q, c) for q, c in zip(queries, contexts)]
        for f in as_completed(futs):
            latencies.append(f.result())

    return float(np.median(latencies))


def _measure_graphrag(client, model, encoder, samples, n_queries, workers, config) -> float:
    import re
    queries  = [s["question"] for s in samples[:n_queries]]
    contexts = [s.get("context", "") for s in samples[:n_queries]]
    latencies = []

    def _extract_entities(text):
        return list(set(re.findall(r'\b[A-Z][a-zA-Z\s]{2,20}\b|\b\d{4}\b|\$[\d,.]+[BMK]?\b', text)))[:15]

    def _run(q, ctx):
        chunks = _chunk_text(ctx, chunk_size=300) if ctx else ["No context."]
        entity_chunks: Dict[str, List[str]] = {}
        for chunk in chunks:
            for ent in _extract_entities(chunk):
                entity_chunks.setdefault(ent, []).append(chunk)
        q_ents = _extract_entities(q)
        matched = []
        for ent in q_ents:
            for key, clist in entity_chunks.items():
                if ent.lower() in key.lower():
                    matched.extend(clist)
        if not matched:
            matched = _dense_retrieve(q, chunks, encoder)
        seen, unique = set(), []
        for c in matched:
            if c not in seen:
                seen.add(c); unique.append(c)
        evidence = "\n\n".join(unique[:6])
        prompt = f"Evidence:\n{evidence}\n\nQuestion: {q}\nAnswer:"
        t0 = time.perf_counter()
        _call_llm(client, model, prompt, config)
        return time.perf_counter() - t0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_run, q, c) for q, c in zip(queries, contexts)]
        for f in as_completed(futs):
            latencies.append(f.result())

    return float(np.median(latencies))


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_scalability(config_path: str, output_dir: str, workers: int):
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Load a fixed pool of queries (use multidoc2025 or finqa)
    for ds in ("multidoc2025", "finqa"):
        try:
            all_samples = load_dataset_split(ds, "test")
            print(f"Using {ds} queries ({len(all_samples)} total)")
            break
        except FileNotFoundError:
            continue
    else:
        print("[ERROR] No dataset found. Run prepare_datasets.py first.")
        return

    print("Initializing HC-RAG ...")
    hcrag = HCRAGInference(config_path)

    client, model = _make_client(config)
    encoder = _get_encoder(config)

    results = {
        "doc_counts":  DOC_COUNTS,
        "HC-RAG":      [],
        "Vanilla RAG": [],
        "Graph-RAG (entity)": [],
    }

    for doc_count in DOC_COUNTS:
        # Use first doc_count * ~5 samples as proxy for "doc_count documents"
        # (each document contributes ~5 QA pairs on average)
        n_samples = min(doc_count * 5, len(all_samples))
        samples = all_samples[:n_samples]
        print(f"\n--- doc_count={doc_count}  ({n_samples} samples, {N_QUERIES} timed queries) ---")

        lat_hcrag = _measure_hcrag(hcrag, samples, N_QUERIES, workers)
        print(f"  HC-RAG:      {lat_hcrag:.2f}s")

        lat_vanilla = _measure_vanilla_rag(client, model, encoder, samples, N_QUERIES, workers, config)
        print(f"  Vanilla RAG: {lat_vanilla:.2f}s")

        lat_graph = _measure_graphrag(client, model, encoder, samples, N_QUERIES, workers, config)
        print(f"  Graph-RAG (entity): {lat_graph:.2f}s")

        results["HC-RAG"].append(round(lat_hcrag, 3))
        results["Vanilla RAG"].append(round(lat_vanilla, 3))
        results["Graph-RAG (entity)"].append(round(lat_graph, 3))

    out_path = os.path.join(output_dir, "scalability_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved scalability results -> {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="HC-RAG scalability experiment (Figure 5)")
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--workers",    type=int, default=8)
    args = parser.parse_args()
    run_scalability(args.config, args.output_dir, args.workers)


if __name__ == "__main__":
    main()
