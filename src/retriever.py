"""
Hierarchical Retrieval with Cross-Modal Fusion
L1: Document-level filtering (metadata + embeddings)
L2: Section-level localization
L3: Semantic unit-level extraction with modality fusion
"""

import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import faiss

from .hierarchical_index import HierarchicalIndex, NodeType, DocumentNode, SectionNode, BaseNode
from .encoders import RetrievalEncoder
from .fusion import AdaptiveFusionNetwork, IntentClassifier, HybridRetrievalFusion, IntentType


class HierarchicalRetriever:
    """
    Three-level hierarchical retriever with query-aware cross-modal fusion
    """
    
    def __init__(self, index: HierarchicalIndex, 
                 encoder: RetrievalEncoder,
                 fusion_network: AdaptiveFusionNetwork,
                 intent_classifier: IntentClassifier,
                 config: Dict[str, Any]):
        self.index = index
        self.encoder = encoder
        self.fusion_network = fusion_network
        self.intent_classifier = intent_classifier
        self.config = config
        
        self.l1_k = config.get("l1_document_k", 5)
        self.l2_k = config.get("l2_section_k", 10)
        self.l3_k = config.get("l3_semantic_k", 20)
        
        # FAISS indices for efficient similarity search (optional)
        self.text_faiss_index = None
        self.table_faiss_index = None
        
        # Cached embeddings
        self.text_embeddings = {}  # node_id -> embedding
        self.table_embeddings = {}
        
    def build_faiss_index(self):
        """Build FAISS index for efficient retrieval"""
        text_embeds = []
        text_ids = []
        
        for node_id, node in self.index.text_chunk_nodes.items():
            if node.embedding is not None:
                text_embeds.append(node.embedding)
                text_ids.append(node_id)
        
        if text_embeds:
            text_embeds = np.stack(text_embeds).astype('float32')
            self.text_faiss_index = faiss.IndexFlatIP(text_embeds.shape[1])
            self.text_faiss_index.add(text_embeds)
            self.text_id_to_idx = {node_id: i for i, node_id in enumerate(text_ids)}
            self.idx_to_text_id = {i: node_id for i, node_id in enumerate(text_ids)}
        
        # Table cells FAISS
        table_embeds = []
        table_ids = []
        for node_id, node in self.index.table_cell_nodes.items():
            if node.embedding is not None:
                table_embeds.append(node.embedding)
                table_ids.append(node_id)
        
        if table_embeds:
            table_embeds = np.stack(table_embeds).astype('float32')
            self.table_faiss_index = faiss.IndexFlatIP(table_embeds.shape[1])
            self.table_faiss_index.add(table_embeds)
            self.table_id_to_idx = {node_id: i for i, node_id in enumerate(table_ids)}
            self.idx_to_table_id = {i: node_id for i, node_id in enumerate(table_ids)}
    
    def retrieve(self, query: str) -> Tuple[List[BaseNode], float, IntentType]:
        """
        Hierarchical retrieval with three levels

        Returns:
            evidence_nodes: list of retrieved semantic units
            fusion_weight: λ value used for fusion
            intent: predicted query intent
        """
        # Encode query
        query_embedding = self.encoder.encode_query(query)

        # Classify intent — supports both old embedding-based and new FinBERT wrapper
        _emb_tensor = torch.from_numpy(query_embedding).float()
        try:
            intent_probs = self.intent_classifier.get_intent_probs(_emb_tensor, query=query)
            intent = self.intent_classifier.predict_intent(_emb_tensor, query=query)
        except TypeError:
            intent_probs = self.intent_classifier.get_intent_probs(_emb_tensor)
            intent = self.intent_classifier.predict_intent(_emb_tensor)

        # L1: Document-level retrieval
        relevant_docs = self._retrieve_documents(query, query_embedding)

        # L2: Section-level retrieval within relevant documents
        relevant_sections = self._retrieve_sections(query, query_embedding, relevant_docs)

        # L3: Semantic unit retrieval with cross-modal fusion
        evidence_nodes, fusion_weight = self._retrieve_semantic_units(
            query, query_embedding, relevant_sections, intent_probs
        )

        # Cross-doc coverage guarantee: every L1 document must have at least one
        # node in the final evidence set.  If a doc is missing, inject its
        # highest-scoring section's best chunk so cross-doc recall is not lost.
        evidence_nodes = self._ensure_doc_coverage(
            evidence_nodes, relevant_docs, relevant_sections, query_embedding
        )

        return evidence_nodes, fusion_weight, intent
    
    # Ticker → common name mapping for S&P 500 companies in the index
    _TICKER_TO_NAME = {
        "aapl": "apple", "msft": "microsoft", "googl": "google", "goog": "google",
        "amzn": "amazon", "meta": "meta", "nvda": "nvidia", "tsla": "tesla",
        "brk": "berkshire", "unh": "unitedhealth", "jnj": "johnson", "jpm": "jpmorgan",
        "v": "visa", "pg": "procter", "ma": "mastercard", "hd": "home depot",
        "cvx": "chevron", "mrk": "merck", "abbv": "abbvie", "pep": "pepsico",
        "ko": "coca", "avgo": "broadcom", "cost": "costco", "wmt": "walmart",
        "dis": "disney", "csco": "cisco", "abt": "abbott", "tmo": "thermo",
        "acn": "accenture", "nke": "nike", "adbe": "adobe", "crm": "salesforce",
        "txn": "texas instruments", "qcom": "qualcomm", "intc": "intel",
        "amd": "advanced micro", "hon": "honeywell", "ups": "united parcel",
        "cat": "caterpillar", "ba": "boeing", "ge": "general electric",
        "xom": "exxon", "cvs": "cvs", "ci": "cigna", "mo": "altria",
        "bac": "bank of america", "wfc": "wells fargo", "c": "citigroup",
        "gs": "goldman", "ms": "morgan stanley", "blk": "blackrock",
        "spgi": "sp global", "mcd": "mcdonald", "sbux": "starbucks",
        "low": "lowes", "tgt": "target", "f": "ford", "gm": "general motors",
        "pfe": "pfizer", "bmy": "bristol", "lly": "eli lilly", "amgn": "amgen",
        "gild": "gilead", "regn": "regeneron", "vrtx": "vertex",
    }

    def _retrieve_documents(self, query: str, query_embedding: np.ndarray) -> List[DocumentNode]:
        """L1: Document-level retrieval using metadata filtering + embedding similarity.

        Metadata signals (ticker, company name, fiscal year, industry) are matched
        against the query. When all metadata scores are zero (generic query with no
        company mention), embedding cosine similarity is used as the sole ranking signal
        so that retrieval never degenerates to arbitrary ordering.
        """
        candidate_docs = list(self.index.doc_nodes.values())
        query_lower = query.lower()

        import re
        year_hits = set(re.findall(r'\b20[2-9]\d\b', query))

        doc_scores = []
        for doc in candidate_docs:
            score = 0.0
            meta = doc.metadata

            ticker  = str(meta.get("company_name", "")).lower()  # stored as ticker
            year    = str(meta.get("fiscal_year", ""))
            industry = str(meta.get("industry", "")).lower()

            # Ticker direct match (e.g. "META" in query)
            if ticker and ticker in query_lower:
                score += 1.5

            # Ticker → full name match
            full_name = self._TICKER_TO_NAME.get(ticker, "")
            if full_name and full_name in query_lower:
                score += 1.2
            for token in full_name.split():
                if len(token) > 3 and token in query_lower:
                    score += 0.4

            # Fiscal year match
            if year and year in year_hits:
                score += 0.8

            # Industry keyword match
            for token in industry.split():
                if len(token) > 4 and token in query_lower:
                    score += 0.3

            # Cross-doc query boost
            if any(kw in query_lower for kw in ("compare", "versus", "vs", "both", "each")):
                score += 0.2

            # Embedding similarity — always computed, weighted higher when metadata score is low
            doc_emb = getattr(doc, "embedding", None)
            if doc_emb is not None:
                sim = float(np.dot(doc_emb, query_embedding) /
                            (np.linalg.norm(doc_emb) * np.linalg.norm(query_embedding) + 1e-9))
                score += sim * 0.5

            doc_scores.append((doc, score))

        # If all metadata scores are 0 (no company/year mention), rely purely on embedding sim
        max_meta_score = max((s for _, s in doc_scores), default=0.0)
        if max_meta_score == 0.0:
            # Re-score using only embedding similarity
            doc_scores = []
            for doc in candidate_docs:
                doc_emb = getattr(doc, "embedding", None)
                if doc_emb is not None:
                    sim = float(np.dot(doc_emb, query_embedding) /
                                (np.linalg.norm(doc_emb) * np.linalg.norm(query_embedding) + 1e-9))
                else:
                    sim = 0.0
                doc_scores.append((doc, sim))

        doc_scores.sort(key=lambda x: x[1], reverse=True)

        # Forced inclusion: any doc whose ticker appears explicitly in the query
        # must be in the result set regardless of rank.  This ensures cross-doc
        # queries (e.g. "compare AEP and ED") always retrieve both target documents.
        import re as _re
        # Match whole-word tickers (upper-case, 1-5 chars) in the query
        query_tickers = set(_re.findall(r'\b[A-Z]{1,5}\b', query))
        forced = []
        remaining = []
        for doc, score in doc_scores:
            ticker = str(doc.metadata.get("company_name", "")).upper()
            if ticker and ticker in query_tickers:
                forced.append((doc, score))
            else:
                remaining.append((doc, score))

        # Fill up to l1_k: forced docs first, then top-scoring remainder
        combined = forced + remaining
        return [doc for doc, _ in combined[:self.l1_k]]

    def _retrieve_sections(self, query: str, query_embedding: np.ndarray,
                           documents: List[DocumentNode]) -> List[SectionNode]:
        """L2: Section-level retrieval using embedding cosine similarity.

        Each section's embedding is computed on demand (and cached on the node) if
        not already present.  Similarity is computed against the query embedding,
        matching the paper's description of L2 localization.

        Per-document quota: each document is guaranteed at least
        ceil(l2_k / n_docs) section slots so that cross-doc queries are not
        dominated by a single large document with many sections.
        """
        import math

        if not documents:
            return []

        # Per-doc quota: spread l2_k evenly, minimum 1 per doc
        n_docs = len(documents)
        quota_per_doc = max(1, math.ceil(self.l2_k / n_docs))

        selected = []
        for doc in documents:
            doc_sections = self.index.get_document_sections(doc.node_id)
            if not doc_sections:
                continue

            sec_scores = []
            for section in doc_sections:
                sec_emb = getattr(section, "embedding", None)

                if sec_emb is None:
                    title = section.metadata.get("title", "")
                    child_text = ""
                    for chunk in self.index.get_section_chunks(section.node_id)[:3]:
                        if hasattr(chunk, "content"):
                            c = chunk.content
                            if isinstance(c, dict):
                                child_text += " " + " ".join(str(v) for v in c.values())
                            elif isinstance(c, str):
                                child_text += " " + c
                    sec_text = (title + " " + child_text).strip()[:512]
                    sec_emb = self.encoder.encode_text_chunk(sec_text)
                    section.embedding = sec_emb

                sim = float(np.dot(sec_emb, query_embedding) /
                            (np.linalg.norm(sec_emb) * np.linalg.norm(query_embedding) + 1e-9))
                sec_scores.append((section, sim))

            sec_scores.sort(key=lambda x: x[1], reverse=True)
            selected.extend(sec_scores[:quota_per_doc])

        # Re-rank the per-doc winners globally and return top l2_k
        selected.sort(key=lambda x: x[1], reverse=True)
        return [sec for sec, _ in selected[:self.l2_k]]
    
    def _ensure_doc_coverage(self, evidence_nodes: List[BaseNode],
                              relevant_docs: List[DocumentNode],
                              relevant_sections: List[SectionNode],
                              query_embedding: np.ndarray) -> List[BaseNode]:
        """Guarantee every L1 document has at least one node in the evidence set.

        For each L1 doc that is absent from evidence_nodes, find its best section
        from relevant_sections and inject that section's highest-scoring chunk.
        This preserves cross-doc recall without disrupting the fusion ranking.
        """
        # Build set of doc_ids already covered in evidence
        def _doc_id_of(node) -> str:
            """Walk reverse_edges two levels up: chunk → section → doc."""
            p1 = self.index.reverse_edges.get(node.node_id, [])
            if not p1:
                return ""
            p2 = self.index.reverse_edges.get(p1[0][0], [])
            return p2[0][0] if p2 else ""

        covered = {_doc_id_of(n) for n in evidence_nodes}
        covered.discard("")

        # Build section → doc_id map for quick lookup
        sec_to_doc = {}
        for sec in relevant_sections:
            p = self.index.reverse_edges.get(sec.node_id, [])
            if p:
                sec_to_doc[sec.node_id] = p[0][0]

        injected = list(evidence_nodes)
        for doc in relevant_docs:
            if doc.node_id in covered:
                continue
            # Find the best section from this doc that is in relevant_sections
            doc_secs = [s for s in relevant_sections
                        if sec_to_doc.get(s.node_id) == doc.node_id]
            if not doc_secs:
                # Fall back to any section of this doc
                doc_secs = self.index.get_document_sections(doc.node_id)
            if not doc_secs:
                continue
            # Pick the section with highest embedding similarity to query
            best_sec = max(
                doc_secs,
                key=lambda s: float(np.dot(
                    getattr(s, "embedding", np.zeros_like(query_embedding)),
                    query_embedding
                ) / (np.linalg.norm(getattr(s, "embedding", query_embedding)) *
                     np.linalg.norm(query_embedding) + 1e-9))
            )
            # Inject the best chunk from that section
            chunks = self.index.get_section_chunks(best_sec.node_id)
            if not chunks:
                continue
            best_chunk = max(
                chunks,
                key=lambda c: float(np.dot(
                    getattr(c, "embedding", np.zeros_like(query_embedding)),
                    query_embedding
                ) / (np.linalg.norm(getattr(c, "embedding", query_embedding)) *
                     np.linalg.norm(query_embedding) + 1e-9))
            )
            injected.append(best_chunk)
            covered.add(doc.node_id)

        return injected

    def _retrieve_semantic_units(self, query: str, query_embedding: np.ndarray,
                                  sections: List[SectionNode],
                                  intent_probs: np.ndarray) -> Tuple[List[BaseNode], float]:
        """L3: Semantic unit retrieval with cross-modal fusion"""
        # Collect text chunks and table cells from sections
        text_chunks = []
        # table cells grouped by section for per-section diversity cap
        table_cells_by_section: Dict[str, List] = {}

        for section in sections:
            chunks = self.index.get_section_chunks(section.node_id)
            sec_tables = []
            for chunk in chunks:
                if chunk.node_type == NodeType.TEXT_CHUNK:
                    text_chunks.append(chunk)
                elif chunk.node_type == NodeType.TABLE_CELL:
                    sec_tables.append(chunk)
            if sec_tables:
                table_cells_by_section[section.node_id] = sec_tables

        # Encode text chunks (use cached if available)
        text_embeds = []
        for chunk in text_chunks:
            if chunk.embedding is not None:
                text_embeds.append(chunk.embedding)
            else:
                emb = self.encoder.encode_text_chunk(chunk.content)
                chunk.embedding = emb
                text_embeds.append(emb)

        # For table cells: score per section, keep top-K per section to prevent
        # a single large table from dominating the candidate pool.
        # This enforces the topology-constrained diversity from the paper.
        cells_per_section = max(self.l3_k, 20)
        table_scores = []
        for sec_id, cells in table_cells_by_section.items():
            cell_embeds = []
            for cell in cells:
                if cell.embedding is not None:
                    cell_embeds.append(cell.embedding)
                else:
                    cell_text = f"{cell.metadata.get('row_header', '')} {cell.metadata.get('col_header', '')} {cell.metadata.get('value', '')}"
                    emb = self.encoder.encode_text_chunk(cell_text)
                    cell.embedding = emb
                    cell_embeds.append(emb)
            if not cell_embeds:
                continue
            cell_embeds_np = np.stack(cell_embeds)
            sims = np.dot(cell_embeds_np, query_embedding)
            sec_scores = sorted(
                [(cells[i], float(sims[i])) for i in range(len(cells))],
                key=lambda x: x[1], reverse=True
            )
            table_scores.extend(sec_scores[:cells_per_section])

        # Compute similarity scores for text
        text_scores = []
        if text_embeds:
            text_embeds_np = np.stack(text_embeds)
            similarities = np.dot(text_embeds_np, query_embedding)
            text_scores = [(text_chunks[i], similarities[i]) for i in range(len(text_chunks))]
            text_scores.sort(key=lambda x: x[1], reverse=True)

        table_scores.sort(key=lambda x: x[1], reverse=True)

        # Take top-k from each modality
        top_text = text_scores[:self.l3_k]
        top_table = table_scores[:self.l3_k]

        # Get fusion weight
        device = next(self.fusion_network.gate.parameters()).device
        query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0).to(device)
        intent_tensor = torch.from_numpy(intent_probs).float().unsqueeze(0).to(device)

        with torch.no_grad():
            gate_input = torch.cat([query_tensor, intent_tensor], dim=-1)
            fusion_weight = self.fusion_network.gate(gate_input).item()

        # Weight and merge results with guaranteed minimum text slots.
        # Even when fusion_weight is near 0 (table-heavy), we always include
        # at least half text chunks so the LLM has narrative context.
        text_weight = fusion_weight
        table_weight = 1 - fusion_weight

        merged = []
        for chunk, score in top_text:
            merged.append((chunk, score * text_weight, "text"))
        for cell, score in top_table:
            merged.append((cell, score * table_weight, "table"))

        merged.sort(key=lambda x: x[1], reverse=True)

        # Guarantee at least half of l3_k slots are text chunks
        min_text = self.l3_k // 2
        result = []
        text_added = 0
        table_added = 0
        for item in merged:
            if len(result) >= self.l3_k:
                break
            if item[2] == "text":
                result.append(item)
                text_added += 1
            elif item[2] == "table":
                result.append(item)
                table_added += 1

        # If not enough text chunks, fill from top_text
        if text_added < min_text:
            existing_ids = {id(item[0]) for item in result}
            for chunk, score in top_text:
                if text_added >= min_text:
                    break
                if id(chunk) not in existing_ids:
                    result.append((chunk, score * text_weight, "text"))
                    text_added += 1

        return [item[0] for item in result[:self.l3_k]], fusion_weight


class ContextBuilder:
    """
    Build context from retrieved evidence for generation.
    Table cells from the same table are reconstructed into a markdown table
    so the LLM can reason over structured data rather than scattered cells.
    """

    def __init__(self, max_text_length: int = 8000):
        self.max_text_length = max_text_length

    def build_context(self, evidence_nodes: List[BaseNode],
                      include_source: bool = True) -> str:
        """Build context string from retrieved nodes.

        Table cells sharing the same table_id are grouped and rendered as a
        markdown table.  Text chunks are rendered as plain paragraphs.
        The order of first appearance in evidence_nodes determines section order.
        """
        from collections import OrderedDict

        # Separate text chunks and table cells; preserve encounter order for tables
        text_parts: List[str] = []
        # table_id -> {col_header -> {row_header -> value}}
        tables: "OrderedDict[str, Dict]" = OrderedDict()
        table_section: Dict[str, str] = {}  # table_id -> section label

        for node in evidence_nodes:
            if node.node_type == NodeType.TEXT_CHUNK:
                content = node.content if isinstance(node.content, str) else str(node.content)
                if content.strip():
                    text_parts.append(content.strip())

            elif node.node_type == NodeType.TABLE_CELL:
                tid = node.metadata.get("table_id", "table")
                row = str(node.metadata.get("row_header", "")).strip()
                col = str(node.metadata.get("col_header", "")).strip()
                val = str(node.metadata.get("value", "")).strip()
                if tid not in tables:
                    tables[tid] = OrderedDict()
                    sec = node.metadata.get("section", node.metadata.get("title", ""))
                    table_section[tid] = sec
                if col not in tables[tid]:
                    tables[tid][col] = OrderedDict()
                tables[tid][col][row] = val

        context_parts: List[str] = list(text_parts)

        # Render each table as markdown
        for tid, col_map in tables.items():
            cols = list(col_map.keys())
            # Collect all row headers in encounter order
            rows_seen: "OrderedDict[str, None]" = OrderedDict()
            for col in cols:
                for row in col_map[col]:
                    rows_seen[row] = None
            row_headers = list(rows_seen.keys())

            if not cols or not row_headers:
                continue

            sec_label = table_section.get(tid, "")
            header = f"[Table: {tid}" + (f" | {sec_label}" if sec_label else "") + "]"

            # Build markdown table: first column is row header
            md_cols = [""] + cols
            sep = "| " + " | ".join(["---"] * len(md_cols)) + " |"
            header_row = "| " + " | ".join(md_cols) + " |"
            rows_md = []
            for rh in row_headers:
                cells = [rh] + [col_map[c].get(rh, "") for c in cols]
                rows_md.append("| " + " | ".join(cells) + " |")

            table_md = "\n".join([header, header_row, sep] + rows_md)
            context_parts.append(table_md)

        context = "\n\n".join(context_parts)
        if len(context) > self.max_text_length:
            context = context[:self.max_text_length] + "..."
        return context

    def _get_parent_section(self, node: BaseNode):
        return None