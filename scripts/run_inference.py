"""
End-to-end inference for HC-RAG
"""

import sys
import os
import yaml
import json
from typing import Dict, Any, List
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_index import HierarchicalIndex, DocumentNode, SectionNode, TextChunkNode, TableCellNode
from src.encoders import TextEncoder, TableEncoder, RetrievalEncoder
from src.fusion import IntentClassifier, AdaptiveFusionNetwork, IntentType
from src.retriever import HierarchicalRetriever, ContextBuilder
from src.generator import ResponseGenerator


class FinBERTIntentClassifier(nn.Module):
    """Fine-tuned FinBERT intent classifier (matches train_intent.py architecture)."""
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


class FinBERTIntentWrapper:
    """
    Wraps FinBERTIntentClassifier to expose the same interface as IntentClassifier
    (predict_intent / get_intent_probs), so the rest of the pipeline is unchanged.
    """
    def __init__(self, model: FinBERTIntentClassifier, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def _encode(self, query: str):
        enc = self.tokenizer(query, max_length=128, padding="max_length",
                             truncation=True, return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    def predict_intent(self, query_embedding=None, query: str = "") -> IntentType:
        ids, mask = self._encode(query)
        with torch.no_grad():
            logits = self.model(ids, mask)
        return IntentType(torch.argmax(logits, dim=-1).item())

    def get_intent_probs(self, query_embedding=None, query: str = "") -> np.ndarray:
        ids, mask = self._encode(query)
        with torch.no_grad():
            logits = self.model(ids, mask)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()[0]

    # Keep duck-typing compatibility with nn.Module usage in retriever
    def eval(self):
        self.model.eval()
        return self

    def to(self, device):
        self.model.to(device)
        return self


class HCRAGInference:
    """
    Complete HC-RAG inference pipeline
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self._init_models()
        self._init_index()
        
        # Build retriever
        self.retriever = HierarchicalRetriever(
            index=self.index,
            encoder=self.retrieval_encoder,
            fusion_network=self.fusion_network,
            intent_classifier=self.intent_classifier,
            config=self.config["index"]
        )
        
        # Context builder and generator
        self.context_builder = ContextBuilder()
        self.generator = ResponseGenerator({
            "model_name":      self.config["models"]["generator"],
            "max_tokens":      self.config["generation"]["max_tokens"],
            "temperature":     self.config["generation"]["temperature"],
            "openai_api_key":  self.config["models"].get("openai_api_key", "") or os.getenv("OPENAI_API_KEY", ""),
            "openai_base_url": self.config["models"].get("openai_base_url", "") or os.getenv("OPENAI_BASE_URL", ""),
            "local_files_only": self.config["models"].get("local_files_only", True),
        })
    
    def _init_models(self):
        """Initialize all models"""
        local_files_only = self.config["models"].get("local_files_only", True)

        # Text encoder
        self.text_encoder = TextEncoder(
            model_name=self.config["models"]["text_encoder"],
            embedding_dim=self.config["alignment"]["embedding_dim"],
            local_files_only=local_files_only,
        )
        self.text_encoder.to(self.device)
        
        # Table encoder
        self.table_encoder = TableEncoder(
            model_name=self.config["models"]["table_encoder"],
            embedding_dim=self.config["alignment"]["embedding_dim"],
            local_files_only=local_files_only,
        )
        self.table_encoder.to(self.device)
        
        # Intent classifier — FinBERT end-to-end fine-tuned
        model_name = self.config["models"]["text_encoder"]
        _tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        _bert_clf = FinBERTIntentClassifier(
            model_name=model_name,
            num_classes=self.config["fusion"]["intent_classes"],
            dropout=self.config["fusion"]["dropout"],
            local_files_only=local_files_only,
        )
        checkpoint_path = os.path.join(
            self.config["paths"]["checkpoint_dir"],
            "intent_best.pt"
        )
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            _bert_clf.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded intent classifier from {checkpoint_path}")
        _bert_clf.to(self.device)
        _bert_clf.eval()
        self.intent_classifier = FinBERTIntentWrapper(_bert_clf, _tokenizer, self.device)
        
        # Fusion network
        self.fusion_network = AdaptiveFusionNetwork(
            embedding_dim=self.config["alignment"]["embedding_dim"],
            hidden_dim=self.config["fusion"]["hidden_dim"],
            num_intents=self.config["fusion"]["intent_classes"]
        )
        self.fusion_network.to(self.device)
        self.fusion_network.eval()
        
        # Retrieval encoder
        self.retrieval_encoder = RetrievalEncoder(self.text_encoder, self.table_encoder)
    
    def _init_index(self):
        """Initialize or load hierarchical index"""
        index_path = os.path.join(self.config["paths"]["index_dir"], "hierarchical_index.pkl")

        self.index = HierarchicalIndex(self.config["index"])

        if os.path.exists(index_path):
            self.index.load(index_path)
            print(f"Loaded index from {index_path}")
            self._ensure_doc_embeddings()
        else:
            print(f"Warning: Index not found at {index_path}. Building demo index...")
            self._build_demo_index()
            os.makedirs(self.config["paths"]["index_dir"], exist_ok=True)
            self.index.save(index_path)

    def _ensure_doc_embeddings(self):
        """Compute document-level embeddings for any DocumentNode missing one.
        Uses the average of its text chunk embeddings as a proxy for the
        document's semantic content. Chunks are matched to documents via
        node_id prefix (e.g. AAPL_2022_* belongs to doc AAPL_2022).
        """
        from src.hierarchical_index import NodeType
        missing = [n for n in self.index.doc_nodes.values()
                   if getattr(n, "embedding", None) is None]
        if not missing:
            return
        print(f"  Computing embeddings for {len(missing)} document nodes...")

        # Build doc_prefix → doc mapping
        # doc node_id format: e.g. "doc_AAPL_2022" or "AAPL_2022"
        doc_prefixes = {}
        for doc in missing:
            meta = doc.metadata
            ticker = meta.get("company_name", "")
            year = meta.get("fiscal_year", "")
            prefix = f"{ticker}_{year}"
            doc_prefixes[prefix] = doc

        # Collect chunks per document
        doc_chunks = {prefix: [] for prefix in doc_prefixes}
        for node in self.index.nodes.values():
            if node.node_type == NodeType.TEXT_CHUNK:
                nid = node.node_id
                for prefix in doc_prefixes:
                    if nid.startswith(prefix):
                        if len(doc_chunks[prefix]) < 20:
                            c = node.content if isinstance(node.content, str) else ""
                            if c.strip():
                                doc_chunks[prefix].append(c[:256])
                        break

        for prefix, doc in doc_prefixes.items():
            texts = doc_chunks.get(prefix, [])
            if not texts:
                meta = doc.metadata
                texts = [f"{meta.get('company_name','')} {meta.get('industry','')} {meta.get('fiscal_year','')}"]
            emb = self.text_encoder(texts)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            doc.embedding = emb.mean(axis=0)
        print(f"  Done computing document embeddings.")
    
    def _build_demo_index(self):
        """Build demo index for testing"""
        # Create a sample document
        doc_node = DocumentNode(
            doc_id="demo_doc_001",
            company_name="Example Corp",
            fiscal_year="2024",
            industry="Technology"
        )
        self.index.add_document(doc_node)
        
        # Create sections
        sections = [
            ("Business Overview", "Item 1", 1),
            ("Risk Factors", "Item 1A", 1),
            ("Financial Statements", "Item 8", 1),
            ("MD&A", "Item 7", 1)
        ]
        
        for title, item, level in sections:
            section = SectionNode(
                section_id=f"sec_{title.replace(' ', '_')}",
                title=title,
                level=level,
                start_pos=0,
                end_pos=1000
            )
            self.index.add_section(section, doc_node.node_id)
            
            # Add sample text chunks
            if title == "Financial Statements":
                # Add table cells
                sample_table = {
                    "header": ["Year", "Revenue", "Net Income"],
                    "rows": [["2023", "$100M", "$15M"], ["2024", "$115M", "$18M"]]
                }
                for row in sample_table["rows"]:
                    for col, header in enumerate(sample_table["header"]):
                        cell = TableCellNode(
                            cell_id=f"cell_{row[0]}_{header}",
                            row_header=row[0],
                            col_header=header,
                            value=row[col],
                            table_id="income_stmt",
                            row_idx=0,
                            col_idx=col
                        )
                        self.index.add_table_cell(cell, section.node_id)
        
        # Add cross-document edge for demo
        self.index.add_cross_doc_edge("demo_doc_001", "demo_doc_001", "self")
    
    def answer(self, query: str) -> Dict[str, Any]:
        """Answer using hierarchical index retrieval (for Multi-Doc-2025 / long-doc datasets)."""
        evidence_nodes, fusion_weight, intent = self.retriever.retrieve(query)
        result = self.generator.generate(query, evidence_nodes, fusion_weight, intent)
        return {
            "query":         query,
            "answer":        result.answer,
            "sources":       result.sources,
            "fusion_weight": result.fusion_weight,
            "intent":        result.intent.name if hasattr(result.intent, 'name') else str(result.intent),
            "confidence":    result.confidence,
            "num_evidence":  len(evidence_nodes),
        }

    def answer_with_context(self, query: str, context: str) -> Dict[str, Any]:
        """
        Answer using inline context (for FinQA / TAT-QA / FinanceBench / DocFinQA).
        Performs dense retrieval over the context chunks, then applies cross-modal
        fusion — same pipeline as the index-based path, just with a local chunk pool
        instead of the hierarchical index.
        """
        import numpy as np
        from src.fusion import IntentType, IntentAwarePromptBuilder
        import torch, torch.nn.functional as F_

        # 1. Classify intent
        ids, mask = self.intent_classifier._encode(query)
        with torch.no_grad():
            logits = self.intent_classifier.model(ids, mask)
            probs  = F_.softmax(logits, dim=-1).cpu().numpy()[0]
        intent = IntentType(int(probs.argmax()))

        # 2. Chunk the context and encode
        # For very long contexts (DocFinQA: ~500K chars), cap at 200K chars
        # to keep chunk count manageable while retaining most content.
        MAX_CONTEXT_CHARS = 200_000
        if len(context) > MAX_CONTEXT_CHARS:
            context = context[:MAX_CONTEXT_CHARS]

        words = context.split()
        chunk_size, overlap = 512, 50
        chunks = []
        i = 0
        while i < len(words):
            chunks.append(" ".join(words[i: i + chunk_size]))
            i += chunk_size - overlap
        if not chunks:
            chunks = [context[:2000]]

        # 3. Dense retrieval over chunks
        # Use top-15 for long-context datasets (DocFinQA), top-5 otherwise
        top_k = 15 if len(chunks) > 100 else 5
        query_emb = self.retrieval_encoder.encode_query(query)
        chunk_embs = []
        for i in range(0, len(chunks), 64):
            batch = chunks[i: i + 64]
            emb = self.retrieval_encoder.text_encoder(batch)
            if isinstance(emb, torch.Tensor):
                emb = emb.detach().cpu().numpy()
            chunk_embs.append(emb)
        chunk_embs = np.concatenate(chunk_embs, axis=0)
        norms = np.linalg.norm(chunk_embs, axis=1, keepdims=True) + 1e-9
        chunk_embs_n = chunk_embs / norms
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-9)
        scores = chunk_embs_n @ q_norm
        top_idx = np.argsort(scores)[::-1][:top_k]
        top_chunks = [chunks[i] for i in top_idx]

        # 4. Fusion weight from intent (same logic as retriever gate)
        device = next(self.fusion_network.gate.parameters()).device
        qt = torch.from_numpy(query_emb).float().unsqueeze(0).to(device)
        it = torch.from_numpy(probs).float().unsqueeze(0).to(device)
        with torch.no_grad():
            fusion_weight = self.fusion_network.gate(
                torch.cat([qt, it], dim=-1)).item()

        # 5. Build evidence string and generate
        # Enforce context budget (same as baselines: 3000 words)
        CONTEXT_BUDGET_WORDS = 3000
        evidence = "\n\n".join(top_chunks)
        words = evidence.split()
        if len(words) > CONTEXT_BUDGET_WORDS:
            evidence = " ".join(words[:CONTEXT_BUDGET_WORDS])

        # Use UNIFIED_PROMPT for fair E2 comparison (same prompt as all baselines)
        UNIFIED_PROMPT = (
            "You are a financial analyst. Answer the question based on the provided evidence.\n"
            "Use the tables and text to extract or calculate the answer. "
            "Be concise and give the final answer directly.\n\n"
            "Evidence:\n{evidence}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
        prompt = UNIFIED_PROMPT.format(evidence=evidence, question=query)
        try:
            answer = self.generator._generate_openai(prompt)
        except Exception as e:
            answer = f"[ERROR] {e}"

        return {
            "query":         query,
            "answer":        answer,
            "sources":       [{"content": c} for c in top_chunks],
            "fusion_weight": fusion_weight,
            "intent":        intent.name,
            "confidence":    float(probs.max()),
            "num_evidence":  len(top_chunks),
        }


def main():
    """Example usage"""
    # Initialize HC-RAG
    hcrag = HCRAGInference("config.yaml")
    
    # Example queries
    queries = [
        "What was the revenue growth from 2023 to 2024?",
        "Calculate the net profit margin for 2024",
        "What are the main risk factors mentioned in the report?"
    ]
    
    for query in queries:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}")
        
        try:
            result = hcrag.answer(query)
            print(f"Intent: {result['intent']}")
            print(f"Fusion Weight (λ): {result['fusion_weight']:.3f}")
            print(f"Answer: {result['answer'][:500]}...")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Sources: {len(result['sources'])} evidence pieces")
            
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
