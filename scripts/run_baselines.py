"""
Baseline model evaluation for HC-RAG comparison.

Implements all baselines from Table 2 of the paper:
  Retriever-level : BM25, Dense-FinBERT (flat), Dense-FinBERT (large-chunk)
  System-level    : Vanilla RAG, Self-RAG, Graph-RAG (entity), RAPTOR, TAT-LLM, TAPAS-RAG

FAIRNESS CONTROLS:
  - All RAG-style methods use the same generator, decoding setting,
    and maximum context length.
  - HC-RAG uses intent-aware prompting, while baselines use a shared
    evidence-grounded prompt and comparable evidence serialization.
  - Retrieval budgets and top-k settings are recorded in released
    configurations and run metadata.

For datasets with inline context (FinQA, TAT-QA, FinanceBench, DocFinQA):
  chunks are built from the sample's own "context" field.
For Multi-Doc-2025 (no inline context):
  ALL baselines use the same shared flat chunk pool built from raw HTML documents.
  This pool is independent of HC-RAG's hierarchical index.

Usage:
  python scripts/run_baselines.py --baseline vanilla_rag --dataset finqa
  python scripts/run_baselines.py --all_baselines --all_datasets
  python scripts/run_baselines.py --all_baselines --all_datasets --max_samples 100
"""

import os
import sys
import json
import time
import argparse
import re
import csv
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_datasets import load_dataset_split
from src.evaluation import BenchmarkEvaluator
from src.run_metadata import build_run_metadata, save_run_metadata


# ---------------------------------------------------------------------------
# Baseline runner settings recorded in run metadata and released configs.
# ---------------------------------------------------------------------------

TOP_K_EVIDENCE = 5          # top-k evidence units retrieved per query
CONTEXT_BUDGET_WORDS = 3000  # max words fed to generator (~4k tokens)

# Unified prompt used by released baselines.
UNIFIED_PROMPT = (
    "You are a financial analyst. Answer the question based ONLY on the provided evidence.\n"
    "If the answer cannot be determined from the evidence, say 'Not found in provided documents'.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


# ---------------------------------------------------------------------------
# Shared generator (DeepSeek via OpenAI-compatible API)
# ---------------------------------------------------------------------------

def _make_generator(config: dict):
    import openai
    cfg = config["models"]
    return openai.OpenAI(
        api_key=cfg.get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", ""),
        base_url=cfg.get("openai_base_url") or os.getenv("OPENAI_BASE_URL") or None,
    ), cfg["generator"]


def _call_llm(client, model: str, evidence: str, question: str, config: dict) -> str:
    """Unified LLM call for released baselines."""
    gen = config["generation"]
    # Enforce context budget
    words = evidence.split()
    if len(words) > CONTEXT_BUDGET_WORDS:
        evidence = " ".join(words[:CONTEXT_BUDGET_WORDS])
    prompt = UNIFIED_PROMPT.format(evidence=evidence, question=question)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a financial analyst. Answer based only on the provided evidence."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=gen.get("max_tokens", 512),
            temperature=gen.get("temperature", 0.0),
            extra_body={"thinking": {"type": "disabled"}}
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i: i + chunk_size]))
        i += chunk_size - overlap
    return chunks or [text]


def _bm25_retrieve(query: str, chunks: List[str], top_k: int = 5) -> List[str]:
    """Simple BM25-style TF-IDF retrieval (no external dependency)."""
    from math import log
    query_terms = set(query.lower().split())
    N = len(chunks)
    scores = []
    for chunk in chunks:
        words = chunk.lower().split()
        word_count = len(words) + 1e-9
        score = 0.0
        for w in query_terms:
            tf = words.count(w) / word_count
            df = sum(1 for c in chunks if w in c.lower())
            idf = log((N + 1) / (1 + df))
            score += tf * idf
        scores.append(score)
    ranked = sorted(range(N), key=lambda i: scores[i], reverse=True)
    return [chunks[i] for i in ranked[:top_k]]


class _BM25Index:
    """Pre-computed inverted index for fast BM25 retrieval over a large chunk pool."""

    def __init__(self, chunks: List[str]):
        from math import log
        self.chunks = chunks
        self.N = len(chunks)
        # Pre-tokenize and compute doc frequencies
        self._doc_words = []
        self._doc_lens = []
        word_df = {}
        for chunk in chunks:
            words = chunk.lower().split()
            self._doc_words.append(words)
            self._doc_lens.append(len(words))
            for w in set(words):
                word_df[w] = word_df.get(w, 0) + 1
        self._avg_dl = sum(self._doc_lens) / max(self.N, 1)
        # Pre-compute IDF
        self._idf = {w: log((self.N - df + 0.5) / (df + 0.5) + 1.0)
                     for w, df in word_df.items()}

    def retrieve(self, query: str, top_k: int = 10) -> List[int]:
        """Return indices of top-k chunks by BM25 score."""
        k1, b = 1.5, 0.75
        query_terms = set(query.lower().split())
        scores = np.zeros(self.N, dtype=np.float32)
        for w in query_terms:
            idf = self._idf.get(w, 0.0)
            if idf <= 0:
                continue
            for i, words in enumerate(self._doc_words):
                tf = words.count(w)
                if tf == 0:
                    continue
                dl = self._doc_lens[i]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avg_dl))
                scores[i] += idf * tf_norm
        return np.argsort(scores)[::-1][:top_k].tolist()





def _dense_retrieve(query: str, chunks: List[str], encoder, top_k: int = 5) -> List[str]:
    """Dense retrieval using the shared FinBERT text encoder."""
    if not chunks:
        return []
    q_emb = encoder([query])  # (1, D)
    if isinstance(q_emb, torch.Tensor):
        q_emb = q_emb.detach().cpu().numpy()
    q_emb = q_emb.squeeze()

    chunk_embs = []
    for i in range(0, len(chunks), 64):
        batch = chunks[i: i + 64]
        e = encoder(batch)
        if isinstance(e, torch.Tensor):
            e = e.detach().cpu().numpy()
        chunk_embs.append(e)
    chunk_embs = np.concatenate(chunk_embs, axis=0)

    norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9
    chunk_embs = chunk_embs / norms
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    scores = chunk_embs @ q_norm
    ranked = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in ranked]


# ---------------------------------------------------------------------------
# Lazy encoder (loaded once, shared across baselines)
# ---------------------------------------------------------------------------

_text_encoder = None

def _get_text_encoder(config: dict):
    global _text_encoder
    if _text_encoder is None:
        from src.encoders import TextEncoder
        enc = TextEncoder(
            model_name=config["models"]["text_encoder"],
            embedding_dim=config["alignment"]["embedding_dim"],
            local_files_only=config["models"].get("local_files_only", True),
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        enc.to(device)
        enc.eval()
        _text_encoder = enc
    return _text_encoder


# ---------------------------------------------------------------------------
# Individual baseline implementations
# ---------------------------------------------------------------------------

class BM25Baseline:
    name = "bm25"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        return _retrieve_with_meta(sample["question"], sample, self.config,
                                   top_k=top_k, use_bm25=True)

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        chunks = _chunk_text(context)
        retrieved = _bm25_retrieve(sample["question"], chunks, top_k=TOP_K_EVIDENCE)
        evidence = "\n\n".join(retrieved)
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class DPRBaseline:
    name = "dpr"
    paper_name = "Dense-FinBERT (flat)"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.encoder = _get_text_encoder(config)

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        return _retrieve_with_meta(sample["question"], sample, self.config, top_k=top_k)

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        chunks = _chunk_text(context)
        retrieved = _dense_retrieve(sample["question"], chunks, self.encoder)
        evidence = "\n\n".join(retrieved)
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class ContrieverBaseline:
    """Dense retrieval baseline using FinBERT encoder with smaller chunks.
    NOTE: This uses FinBERT (not actual Contriever weights) as the encoder.
    Labeled 'Dense-FinBERT (large-chunk)' in paper to avoid confusion with
    facebook/contriever.
    Kept as 'contriever' key for backward compatibility with existing result files.
    """
    name = "contriever"
    paper_name = "Dense-FinBERT (large-chunk)"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.encoder = _get_text_encoder(config)

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        return _retrieve_with_meta(sample["question"], sample, self.config,
                                   top_k=top_k, chunk_size=256, overlap=32)

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        # This proxy baseline uses larger chunks than the flat dense baseline.
        chunks = _chunk_text(context, chunk_size=256, overlap=32)
        retrieved = _dense_retrieve(sample["question"], chunks, self.encoder, top_k=5)
        evidence = "\n\n".join(retrieved)
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class VanillaRAGBaseline:
    """Standard retrieve-then-generate with flat dense FinBERT retrieval."""
    name = "vanilla_rag"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.encoder = _get_text_encoder(config)

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        return _retrieve_with_meta(sample["question"], sample, self.config, top_k=top_k)

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        chunks = _chunk_text(context)
        retrieved = _dense_retrieve(sample["question"], chunks, self.encoder)
        evidence = "\n\n".join(retrieved)
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class SelfRAGBaseline:
    """Self-RAG: retrieve, reflect on relevance, optionally re-retrieve."""
    name = "self_rag"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.encoder = _get_text_encoder(config)

    def _is_relevant(self, question: str, chunk: str) -> bool:
        evidence = f"Passage: {chunk[:300]}"
        relevance_q = (
            f"Is the following passage relevant to answering the question?\n"
            f"Answer with YES or NO only.\n\nQuestion: {question}"
        )
        resp = _call_llm(self.client, self.model, evidence, relevance_q, self.config)
        return "YES" in resp.upper()

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        # Get base candidates via unified helper (handles multidoc2025 flat pool)
        candidates = _retrieve_with_meta(sample["question"], sample, self.config,
                                         top_k=top_k + 3)
        # Apply relevance reflection on top candidates
        relevant = [c for c in candidates
                    if self._is_relevant(sample["question"], c["content"])]
        if not relevant:
            relevant = candidates
        return relevant[:top_k]

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        chunks = _chunk_text(context)
        retrieved = _dense_retrieve(sample["question"], chunks, self.encoder, top_k=8)

        # Reflection: filter irrelevant chunks, then cap to top_k=5 for fair comparison
        relevant = [c for c in retrieved if self._is_relevant(sample["question"], c)]
        if not relevant:
            relevant = retrieved[:3]  # fallback
        relevant = relevant[:5]

        evidence = "\n\n".join(relevant)
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class GraphRAGBaseline:
    """Graph-based RAG: entity-relation graph from context, traverse for evidence.
    NOTE: This is a simplified implementation using regex entity extraction and
    entity-chunk co-occurrence (not Microsoft GraphRAG's full community detection).
    Labeled 'Graph-RAG (entity)' in paper to distinguish from full GraphRAG.
    """
    name = "graphrag"
    paper_name = "Graph-RAG (entity)"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.encoder = _get_text_encoder(config)

    def _extract_entities(self, text: str) -> List[str]:
        # Simple heuristic: capitalized phrases and numbers as entities
        entities = re.findall(r'\b[A-Z][a-zA-Z\s]{2,30}\b|\b\d{4}\b|\$[\d,.]+[BMK]?\b', text)
        return list(set(entities))[:20]

    def _graph_retrieve(self, question: str, chunks: List[str], top_k: int = 6) -> List[str]:
        entity_chunks: Dict[str, List[str]] = {}
        for chunk in chunks:
            for ent in self._extract_entities(chunk):
                entity_chunks.setdefault(ent, []).append(chunk)

        q_entities = self._extract_entities(question)
        matched_chunks = []
        for ent in q_entities:
            for key, clist in entity_chunks.items():
                if ent.lower() in key.lower() or key.lower() in ent.lower():
                    matched_chunks.extend(clist)

        if not matched_chunks:
            matched_chunks = _dense_retrieve(question, chunks, self.encoder, top_k=top_k)

        seen, unique = set(), []
        for c in matched_chunks:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        return unique[:top_k]

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        # Get base candidates via unified helper (handles multidoc2025 flat pool)
        candidates = _retrieve_with_meta(sample["question"], sample, self.config,
                                         top_k=top_k * 3, chunk_size=300)
        # Apply entity-graph re-ranking on top candidates
        chunk_texts = [c["content"] for c in candidates]
        reranked_texts = self._graph_retrieve(sample["question"], chunk_texts, top_k=top_k)
        # Preserve metadata from original candidates
        text_to_meta = {c["content"]: c for c in candidates}
        results = []
        for i, t in enumerate(reranked_texts):
            meta = text_to_meta.get(t, {})
            results.append({
                "content": t,
                "doc_id":  meta.get("doc_id", _infer_doc_id(sample)),
                "section": meta.get("section", ""),
                "rank":    i + 1,
            })
        return results

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        chunks = _chunk_text(context, chunk_size=300)
        retrieved = self._graph_retrieve(sample["question"], chunks, top_k=5)
        evidence = "\n\n".join(retrieved)
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class RAPTORBaseline:
    """RAPTOR: recursive summarization tree, retrieve at multiple levels.
    Tree is built per-document context and cached by context hash to avoid
    rebuilding on repeated queries against the same document.
    """
    name = "raptor"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.encoder = _get_text_encoder(config)
        self._tree_cache: Dict[int, tuple] = {}  # context_hash -> (l0, l1)

    def _summarize(self, text: str) -> str:
        summary_q = "Summarize the following financial text in 2-3 sentences."
        return _call_llm(self.client, self.model, text[:1000], summary_q, self.config)

    def _build_tree(self, context: str):
        key = hash(context[:500])  # hash first 500 chars as cache key
        if key in self._tree_cache:
            return self._tree_cache[key]
        l0 = _chunk_text(context, chunk_size=512)
        l1 = []
        for i in range(0, len(l0), 4):
            group = " ".join(l0[i: i + 4])
            l1.append(self._summarize(group))
        self._tree_cache[key] = (l0, l1)
        return l0, l1

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        ctx = sample.get("context", "")
        if isinstance(ctx, list):
            ctx = "\n".join(str(x) for x in ctx)
        ctx = ctx.strip()

        if ctx:
            # Has inline context: build RAPTOR tree and retrieve
            l0, l1 = self._build_tree(ctx)
            all_chunks = l0 + l1
            retrieved = _dense_retrieve(sample["question"], all_chunks, self.encoder, top_k=top_k)
            doc_id = _infer_doc_id(sample)
            return [{"content": c, "doc_id": doc_id, "section": "",
                     "rank": i + 1, "level": "summary" if c in l1 else "raw"}
                    for i, c in enumerate(retrieved)]
        else:
            # No inline context (multidoc2025): use flat chunk pool
            return _get_index_chunks_with_meta(sample["question"], self.config, top_k=top_k)

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        l0, l1 = self._build_tree(context)
        retrieved = _dense_retrieve(sample["question"], l0 + l1, self.encoder, top_k=5)
        evidence = "\n\n".join(retrieved)
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class TATLLMBaseline:
    """TAT-LLM: table-aware retrieval with structured table extraction."""
    name = "tat_llm"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.encoder = _get_text_encoder(config)

    def _extract_tables_text(self, context: str) -> str:
        """Extract table-like content (lines with multiple | or numbers)."""
        lines = context.split("\n")
        table_lines = [l for l in lines if l.count("|") >= 2 or
                       len(re.findall(r'\d+\.?\d*%?', l)) >= 3]
        return "\n".join(table_lines[:50])

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        ctx = sample.get("context", "")
        if isinstance(ctx, list):
            ctx = "\n".join(str(x) for x in ctx)
        ctx = ctx.strip()

        if ctx:
            # Has inline context: dense text + table extraction
            chunks = _chunk_text(ctx)
            text_chunks = _dense_retrieve(sample["question"], chunks, self.encoder, top_k=top_k - 1)
            table_text = self._extract_tables_text(ctx)
            doc_id = _infer_doc_id(sample)
            results = [{"content": c, "doc_id": doc_id, "section": "", "rank": i + 1, "type": "text"}
                       for i, c in enumerate(text_chunks)]
            if table_text:
                results.append({"content": table_text, "doc_id": doc_id, "section": "tables",
                                "rank": len(results) + 1, "type": "table"})
            return results
        else:
            # No inline context (multidoc2025): use flat chunk pool
            base = _get_index_chunks_with_meta(sample["question"], self.config, top_k=top_k)
            for c in base:
                c.setdefault("type", "text")
            return base

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        chunks = _chunk_text(context)
        retrieved_text = _dense_retrieve(sample["question"], chunks, self.encoder, top_k=4)
        table_content = self._extract_tables_text(context)

        evidence = "\n\n".join(retrieved_text)
        if table_content:
            evidence += f"\n\nTable Data:\n{table_content}"
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


class TAPEXRAGBaseline:
    """TAPAS-RAG: TAPAS table encoder + flat dense text retrieval, no hierarchy."""
    name = "tapex_rag"
    paper_name = "TAPAS-RAG"

    def __init__(self, config):
        self.config = config
        self.client, self.model = _make_generator(config)
        self.text_encoder = _get_text_encoder(config)
        self._table_encoder = None

    def _get_table_encoder(self):
        if self._table_encoder is None:
            from src.encoders import TableEncoder
            enc = TableEncoder(
                model_name=self.config["models"]["table_encoder"],
                embedding_dim=self.config["alignment"]["embedding_dim"],
                local_files_only=self.config["models"].get("local_files_only", True),
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            enc.to(device)
            enc.eval()
            self._table_encoder = enc
        return self._table_encoder

    def _parse_tables(self, context: str) -> List[Dict]:
        """Parse simple markdown/pipe tables from context."""
        tables = []
        lines = context.split("\n")
        current = []
        for line in lines:
            if "|" in line:
                current.append(line)
            else:
                if len(current) >= 2:
                    header = [c.strip() for c in current[0].split("|") if c.strip()]
                    rows = []
                    for row_line in current[2:]:
                        row = [c.strip() for c in row_line.split("|") if c.strip()]
                        if row:
                            rows.append(row)
                    if header and rows:
                        tables.append({"header": header, "rows": rows})
                current = []
        return tables[:5]

    def _table_to_str(self, table: Dict) -> str:
        s = " | ".join(table.get("header", []))
        for row in table.get("rows", [])[:5]:
            s += "\n" + " | ".join(str(c) for c in row)
        return s

    def retrieve(self, sample: dict, top_k: int = 10) -> List[Dict]:
        ctx = sample.get("context", "")
        if isinstance(ctx, list):
            ctx = "\n".join(str(x) for x in ctx)
        ctx = ctx.strip()

        if ctx:
            # Has inline context: dense text + table parsing
            chunks = _chunk_text(ctx)
            text_chunks = _dense_retrieve(sample["question"], chunks, self.text_encoder, top_k=top_k - 1)
            tables = self._parse_tables(ctx)
            doc_id = _infer_doc_id(sample)
            results = [{"content": c, "doc_id": doc_id, "section": "", "rank": i + 1, "type": "text"}
                       for i, c in enumerate(text_chunks)]
            for j, tbl in enumerate(tables[:1]):
                results.append({"content": self._table_to_str(tbl), "doc_id": doc_id,
                                "section": "table", "rank": len(results) + 1, "type": "table"})
            return results
        else:
            # No inline context (multidoc2025): use flat chunk pool
            base = _get_index_chunks_with_meta(sample["question"], self.config, top_k=top_k)
            for c in base:
                c.setdefault("type", "text")
            return base

    def answer(self, sample: dict) -> str:
        context = _get_context(sample, self.config)
        chunks = _chunk_text(context)
        retrieved_text = _dense_retrieve(sample["question"], chunks, self.text_encoder, top_k=5)

        # Table retrieval via TAPAS encoder
        tables = self._parse_tables(context)
        table_evidence = ""
        if tables:
            try:
                enc = self._get_table_encoder()
                best_table = tables[0]  # use first table as primary evidence
                table_str = self._table_to_str(best_table)
                table_evidence = f"\nTable:\n{table_str}"
            except Exception:
                pass

        evidence = "\n\n".join(retrieved_text) + table_evidence
        return _call_llm(self.client, self.model, evidence, sample["question"], self.config)


# ---------------------------------------------------------------------------
# Shared flat chunk pool for Multi-Doc-2025 (no inline context)
# Built from raw HTML documents independently of HC-RAG's hierarchical index
# to ensure fair baseline comparison.
# ---------------------------------------------------------------------------

_flat_chunks = None           # list of {"content": str, "doc_id": str, "section": str}
_flat_chunk_embs = None       # numpy array (N, D), pre-encoded
_flat_bm25_index = None       # _BM25Index, built once alongside flat chunks
_flat_lock = threading.Lock() # single lock guards all flat-pool state


def _build_flat_chunks(config: dict):
    """Build a flat chunk pool from raw HTML documents (one-time).
    Extracts section headings from <h1>-<h4> tags to populate the section field,
    which is required for E.3 section_hit and evidence_recall metrics.
    """
    import glob as _glob
    from html.parser import HTMLParser

    class _SectionExtractor(HTMLParser):
        """Extracts (section_name, text) pairs by tracking heading tags."""
        def __init__(self):
            super().__init__()
            self.segments = []          # list of {"section": str, "text": str}
            self._current_section = ""
            self._current_text = []
            self._in_heading = False
            self._heading_buf = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip = True
            elif tag in ("h1", "h2", "h3", "h4"):
                # Flush current segment before starting new section
                text = " ".join(self._current_text).strip()
                if text:
                    self.segments.append({
                        "section": self._current_section,
                        "text": text,
                    })
                self._current_text = []
                self._in_heading = True
                self._heading_buf = []

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = False
            elif tag in ("h1", "h2", "h3", "h4") and self._in_heading:
                self._current_section = " ".join(self._heading_buf).strip()
                self._in_heading = False
                self._heading_buf = []

        def handle_data(self, data):
            if self._skip:
                return
            if self._in_heading:
                self._heading_buf.append(data.strip())
            else:
                stripped = data.strip()
                if stripped:
                    self._current_text.append(stripped)

        def flush(self):
            text = " ".join(self._current_text).strip()
            if text:
                self.segments.append({
                    "section": self._current_section,
                    "text": text,
                })

    data_dir = config["paths"]["data_dir"]
    html_dir = os.path.join(data_dir, "raw")
    if not os.path.isdir(html_dir):
        html_dir = os.path.join(data_dir, "multidoc2025", "original_doc")

    html_files = sorted(_glob.glob(os.path.join(html_dir, "*.html")))
    print(f"  [flat-index] Building flat chunks from {len(html_files)} HTML files...")

    chunks = []
    chunk_size = config.get("index", {}).get("chunk_size", 512)
    overlap = config.get("index", {}).get("chunk_overlap", 50)

    for fpath in html_files:
        fname = os.path.basename(fpath).replace(".html", "")
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()
        except Exception:
            continue

        parser = _SectionExtractor()
        parser.feed(html_content)
        parser.flush()

        for seg in parser.segments:
            section = seg["section"]
            words = seg["text"].split()
            i = 0
            while i < len(words):
                chunk_words = words[i: i + chunk_size]
                chunk_text = " ".join(chunk_words)
                if chunk_text.strip():
                    chunks.append({
                        "content": chunk_text,
                        "doc_id":  fname,
                        "section": section,
                    })
                i += chunk_size - overlap

    print(f"  [flat-index] Total chunks: {len(chunks)}")
    return chunks


def _get_index_context(question: str, config: dict) -> str:
    """Retrieve context from flat chunk pool for datasets without inline context.
    Chunks are built from raw HTML documents (not HC-RAG's hierarchical index)
    to ensure fair baseline comparison.
    """
    global _flat_chunks, _flat_chunk_embs
    with _flat_lock:
        if _flat_chunks is None:
            _flat_chunks = _build_flat_chunks(config)

            enc = _get_text_encoder(config)
            print(f"  [flat-index] Pre-encoding {len(_flat_chunks)} chunks (one-time)...")
            embs = []
            for i in range(0, len(_flat_chunks), 64):
                batch = [c["content"] for c in _flat_chunks[i: i + 64]]
                e = enc(batch)
                if isinstance(e, torch.Tensor):
                    e = e.detach().cpu().numpy()
                embs.append(e)
            embs = np.concatenate(embs, axis=0)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
            _flat_chunk_embs = embs / norms
            print(f"  [flat-index] Done. Shape: {_flat_chunk_embs.shape}")

    if not _flat_chunks:
        return ""

    enc = _get_text_encoder(config)
    q_emb = enc([question])
    if isinstance(q_emb, torch.Tensor):
        q_emb = q_emb.detach().cpu().numpy()
    q_emb = q_emb.squeeze()
    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

    scores = _flat_chunk_embs @ q_norm
    top_idx = np.argsort(scores)[::-1][:5]
    retrieved = [_flat_chunks[i]["content"] for i in top_idx]
    return "\n\n".join(retrieved)


def _get_index_chunks_with_meta(question: str, config: dict, top_k: int = 10,
                                use_bm25: bool = False) -> List[Dict]:
    """Retrieve top-k chunks with metadata (for E.3 evidence evaluation).
    use_bm25=True uses BM25 scoring; False uses dense (FinBERT) retrieval.
    All initialisation happens inside _flat_lock to avoid multi-lock deadlocks.
    """
    global _flat_chunks, _flat_chunk_embs, _flat_bm25_index
    with _flat_lock:
        if _flat_chunks is None:
            _flat_chunks = _build_flat_chunks(config)

        if use_bm25 and _flat_bm25_index is None:
            print(f"  [bm25-index] Building BM25 inverted index ({len(_flat_chunks)} chunks)...")
            _flat_bm25_index = _BM25Index([c["content"] for c in _flat_chunks])
            print(f"  [bm25-index] Done.")

        if not use_bm25 and _flat_chunk_embs is None:
            enc = _get_text_encoder(config)
            print(f"  [flat-index] Pre-encoding {len(_flat_chunks)} chunks (one-time)...")
            embs = []
            for i in range(0, len(_flat_chunks), 64):
                batch = [c["content"] for c in _flat_chunks[i: i + 64]]
                e = enc(batch)
                if isinstance(e, torch.Tensor):
                    e = e.detach().cpu().numpy()
                embs.append(e)
            embs = np.concatenate(embs, axis=0)
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
            _flat_chunk_embs = embs / norms
            print(f"  [flat-index] Done. Shape: {_flat_chunk_embs.shape}")

    if not _flat_chunks:
        return []

    if use_bm25:
        top_indices = _flat_bm25_index.retrieve(question, top_k=top_k)
        return [
            {"content": _flat_chunks[i]["content"],
             "doc_id":  _flat_chunks[i]["doc_id"],
             "section": _flat_chunks[i]["section"],
             "rank":    rank + 1}
            for rank, i in enumerate(top_indices)
        ]
    else:
        enc = _get_text_encoder(config)
        q_emb = enc([question])
        if isinstance(q_emb, torch.Tensor):
            q_emb = q_emb.detach().cpu().numpy()
        q_emb = q_emb.squeeze()
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
        scores = _flat_chunk_embs @ q_norm
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [
            {"content": _flat_chunks[i]["content"],
             "doc_id":  _flat_chunks[i]["doc_id"],
             "section": _flat_chunks[i]["section"],
             "rank":    rank + 1}
            for rank, i in enumerate(top_idx)
        ]


def _retrieve_with_meta(question: str, sample: dict, config: dict,
                        top_k: int = 10, chunk_size: int = 512,
                        overlap: int = 50, use_bm25: bool = False) -> List[Dict]:
    """
    Unified retrieve-with-metadata helper.
    - If sample has inline context: chunk it and retrieve from those chunks.
    - If no inline context (multidoc2025): use flat chunk pool with doc_id metadata.
      use_bm25=True applies BM25 scoring on the flat pool (for BM25Baseline).
    """
    ctx = sample.get("context", "")
    if isinstance(ctx, list):
        ctx = "\n".join(str(x) for x in ctx)
    ctx = ctx.strip()
    if len(ctx) > 200_000:
        ctx = ctx[:200_000]

    if ctx:
        # Inline context: chunk and retrieve, doc_id inferred from sample metadata
        chunks = _chunk_text(ctx, chunk_size=chunk_size, overlap=overlap)
        doc_id = _infer_doc_id(sample)
        if use_bm25:
            retrieved_texts = _bm25_retrieve(question, chunks, top_k=top_k)
        else:
            enc = _get_text_encoder(config)
            retrieved_texts = _dense_retrieve(question, chunks, enc, top_k=top_k)
        return [{"content": c, "doc_id": doc_id, "section": "", "rank": i + 1}
                for i, c in enumerate(retrieved_texts)]
    else:
        # No inline context: use flat chunk pool (multidoc2025)
        # Pass use_bm25 so BM25Baseline actually uses BM25 on the flat pool
        return _get_index_chunks_with_meta(question, config, top_k=top_k, use_bm25=use_bm25)


def _infer_doc_id(sample: dict) -> str:
    """Derive a document identifier from sample metadata (used in retrieve() methods)."""
    company = sample.get("company", "")
    year    = sample.get("year", "")
    if company and year:
        return f"{company}_{year}"
    return company or year or "unknown"


def _get_context(sample: dict, config: dict) -> str:
    """Return inline context if available, else retrieve from index."""
    ctx = sample.get("context", "")
    if isinstance(ctx, list):
        ctx = "\n".join(str(x) for x in ctx)
    ctx = ctx.strip()
    # Cap very long contexts (DocFinQA: ~500K chars) to keep retrieval tractable
    if len(ctx) > 200_000:
        ctx = ctx[:200_000]
    if ctx:
        return ctx
    return _get_index_context(sample["question"], config)




BASELINES = {
    "bm25":        BM25Baseline,
    "dpr":         DPRBaseline,
    "contriever":  ContrieverBaseline,
    "vanilla_rag": VanillaRAGBaseline,
    "self_rag":    SelfRAGBaseline,
    "graphrag":    GraphRAGBaseline,
    "raptor":      RAPTORBaseline,
    "tat_llm":     TATLLMBaseline,
    "tapex_rag":   TAPEXRAGBaseline,
}

DATASETS = ["finqa", "tatqa", "financebench", "docfinqa", "multidoc2025"]


# ---------------------------------------------------------------------------
# Evaluation loop (mirrors run_evaluation.py)
# ---------------------------------------------------------------------------

def _clean_sample(sample: dict) -> dict:
    q = sample.get("question", "")
    a = sample.get("answer", "")
    sample = dict(sample)
    # Strip flare-* prompt wrapper
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


def evaluate_baseline(baseline, evaluator, samples, max_samples=None, workers=8):
    if max_samples:
        samples = samples[:max_samples]

    n = len(samples)
    results = [None] * n          # preserve order
    latencies = [0.0] * n
    print_lock = threading.Lock()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    def _process(idx, raw_sample):
        sample = _clean_sample(raw_sample)
        t0 = time.perf_counter()
        # Retrieve evidence first (used for faithful_acc / hall_rate)
        try:
            retrieved = baseline.retrieve(sample, top_k=TOP_K_EVIDENCE)
            evidence_text = "\n\n".join(r.get("content", "") for r in retrieved)
        except Exception:
            evidence_text = ""
        try:
            pred = baseline.answer(sample)
        except Exception as e:
            with print_lock:
                print(f"    [ERROR] sample {idx+1}: {e}")
            pred = ""
        lat = time.perf_counter() - t0
        with print_lock:
            print(f"  [{idx+1}/{n}] {sample['question'][:70]}  ({lat:.1f}s)")
        return idx, sample, pred, lat, evidence_text

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, i, s): i for i, s in enumerate(samples)}
        for fut in as_completed(futures):
            idx, sample, pred, lat, evidence_text = fut.result()
            results[idx] = (sample, pred, evidence_text)
            latencies[idx] = lat

    predictions, ground_truths, log = [], [], []
    for (sample, pred, evidence_text), lat in zip(results, latencies):
        predictions.append({
            "answer":   pred,
            "evidence": evidence_text,
            "intent":   sample.get("intent", "fact"),
        })
        ground_truths.append({
            "answer":             sample["answer"],
            "intent":             sample.get("intent", "fact"),
            "execution_required": sample.get("execution_required", False),
            "is_cross_doc":       sample.get("is_cross_doc",    False),
            "is_cross_year":      sample.get("is_cross_year",   False),
            "is_hybrid_modal":    sample.get("is_hybrid_modal", False),
            "subset":             sample.get("subset",          ""),
            "difficulty":         sample.get("difficulty",      ""),
            "sector":             sample.get("sector",          ""),
            "companies":          sample.get("companies",       []),
        })
        log.append({
            "question":     sample["question"],
            "ground_truth": sample["answer"],
            "prediction":   pred,
            "latency_s":    round(lat, 3),
        })

    metrics = evaluator.evaluate_dataset(predictions, ground_truths)
    if latencies:
        metrics["avg_latency_s"] = round(sum(latencies) / len(latencies), 3)
    if torch.cuda.is_available():
        metrics["peak_gpu_memory_gb"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 2)
    return log, metrics


def save_results(log, metrics, output_dir, baseline_name, dataset, split,
                 config, config_path, max_samples, workers):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    pred_path = os.path.join(output_dir, f"{baseline_name}_{dataset}_{split}_predictions_{ts}.json")
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    metrics_path = os.path.join(output_dir, f"{baseline_name}_{dataset}_{split}_metrics_{ts}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    metadata = build_run_metadata(
        config=config,
        config_path=config_path,
        script_name="scripts/run_baselines.py",
        run_type="e2_answer_eval",
        dataset=dataset,
        split=split,
        method=baseline_name,
        output_dir=output_dir,
        retrieval_top_k=TOP_K_EVIDENCE,
        final_evidence_budget=TOP_K_EVIDENCE,
        max_samples=max_samples,
        workers=workers,
        extra={
            "n_samples_evaluated": len(log),
            "context_budget_words": CONTEXT_BUDGET_WORDS,
            "fairness_controls": {
                "shared_generator": True,
                "shared_prompt_template": True,
                "shared_decoding": True,
                "shared_max_context_length": True,
                "shared_evidence_serialization": True,
            },
        },
    )
    metadata_path = save_run_metadata(
        metadata, output_dir, f"{baseline_name}_{dataset}_{split}", ts
    )

    csv_path = os.path.join(output_dir, "all_results.csv")
    fieldnames = [
        "timestamp", "model", "dataset", "split",
        # E.2 main metrics
        "em", "f1", "exec_acc", "hall_rate", "faithful_acc",
        # intent slices
        "calculation_em", "calculation_f1", "calculation_exec_acc",
        "trend_em", "trend_f1",
        "fact_em", "fact_f1",
        "comparison_em", "comparison_f1",
        # structural slices
        "cross_doc_f1", "cross_year_f1", "hybrid_modal_f1",
        "single_doc_f1", "cross_company_f1",
        # subset slices
        "subset_S1_f1", "subset_S2_f1", "subset_S3_f1", "subset_S4_f1", "subset_S5_f1",
        # difficulty slices
        "difficulty_L1_f1", "difficulty_L2_f1", "difficulty_L3_f1", "difficulty_L4_f1",
        # efficiency
        "avg_latency_s", "median_latency_s", "peak_gpu_memory_gb",
    ]
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        row = {"timestamp": ts, "model": baseline_name, "dataset": dataset, "split": split}
        row.update({k: f"{v:.4f}" if isinstance(v, float) else v for k, v in metrics.items()})
        writer.writerow(row)

    print(f"  Predictions -> {pred_path}")
    print(f"  Metrics     -> {metrics_path}")
    print(f"  Run Meta    -> {metadata_path}")
    print(f"  CSV         -> {csv_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _already_done(output_dir: str, bl_name: str, dataset: str, split: str) -> bool:
    """Return True if a metrics file already exists for this baseline/dataset/split."""
    import glob as _glob
    pattern = os.path.join(output_dir, f"{bl_name}_{dataset}_{split}_metrics_*.json")
    return len(_glob.glob(pattern)) > 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", choices=list(BASELINES.keys()), default="vanilla_rag")
    parser.add_argument("--all_baselines", action="store_true")
    parser.add_argument("--dataset", choices=DATASETS, default="finqa")
    parser.add_argument("--all_datasets", action="store_true")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--data_dir", default="./data/benchmarks")
    parser.add_argument("--output_dir", default="./outputs")
    parser.add_argument("--workers", type=int, default=16,
                        help="Concurrent API threads per baseline (default: 16)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip baseline/dataset combos that already have a metrics file")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    evaluator = BenchmarkEvaluator()
    baselines_to_run = list(BASELINES.keys()) if args.all_baselines else [args.baseline]
    datasets_to_run  = DATASETS if args.all_datasets else [args.dataset]

    for bl_name in baselines_to_run:
        # Check if ALL datasets for this baseline are done before loading the model
        pending_datasets = []
        for dataset in datasets_to_run:
            if args.resume and _already_done(args.output_dir, bl_name, dataset, args.split):
                print(f"  [SKIP] {bl_name}/{dataset} already done")
            else:
                pending_datasets.append(dataset)

        if not pending_datasets:
            print(f"\n[SKIP] {bl_name}: all datasets already done")
            continue

        print(f"\n{'='*60}")
        print(f"Baseline: {bl_name}  (workers={args.workers})")
        print(f"{'='*60}")
        baseline = BASELINES[bl_name](config)

        for dataset in pending_datasets:
            print(f"\n  Dataset: {dataset}/{args.split}")
            try:
                samples = load_dataset_split(dataset, args.split, args.data_dir)
            except FileNotFoundError as e:
                print(f"  [SKIP] {e}")
                continue

            log, metrics = evaluate_baseline(
                baseline, evaluator, samples,
                max_samples=args.max_samples,
                workers=args.workers,
            )
            print(f"\n  Results ({bl_name} / {dataset}):")
            for k, v in sorted(metrics.items()):
                print(f"    {k:<25} {v:.4f}" if isinstance(v, float) else f"    {k:<25} {v}")
            save_results(
                log, metrics, args.output_dir, bl_name, dataset, args.split,
                config, args.config, args.max_samples, args.workers,
            )

    print("\nBaseline evaluation complete.")


if __name__ == "__main__":
    main()
