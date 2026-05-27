#!/usr/bin/env python3
"""
Demo script for HC-RAG framework
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_index import HierarchicalIndex, DocumentNode, SectionNode, TextChunkNode, TableCellNode
from src.encoders import TextEncoder, TableEncoder, RetrievalEncoder
from src.fusion import IntentClassifier, AdaptiveFusionNetwork
from src.retriever import HierarchicalRetriever
from src.generator import ResponseGenerator
import torch
import numpy as np


def demo_basic_components():
    """Demonstrate basic component functionality without heavy models"""
    print("=" * 60)
    print("HC-RAG Basic Demo")
    print("=" * 60)
    
    # 1. Build a small hierarchical index
    print("\n1. Building Hierarchical Index...")
    index = HierarchicalIndex({})
    
    # Add document
    doc = DocumentNode("demo_doc", "Demo Corp", "2024", "Technology")
    index.add_document(doc)
    
    # Add sections
    sec1 = SectionNode("sec_financial", "Financial Statements", 1, 0, 500)
    sec2 = SectionNode("sec_mda", "Management Discussion", 1, 500, 1000)
    index.add_section(sec1, "demo_doc")
    index.add_section(sec2, "demo_doc")
    
    # Add text chunks
    chunk1 = TextChunkNode("chunk1", "Revenue increased by 15% to $115M.", 0, 50)
    chunk2 = TextChunkNode("chunk2", "Gross margin improved to 45% from 42%.", 50, 100)
    index.add_text_chunk(chunk1, "sec_financial")
    index.add_text_chunk(chunk2, "sec_financial")
    
    # Add table cells
    cell1 = TableCellNode("cell_rev", "2024", "Revenue", "$115M", "table1", 0, 1)
    cell2 = TableCellNode("cell_gm", "2024", "Gross Margin", "45%", "table1", 1, 1)
    index.add_table_cell(cell1, "sec_financial")
    index.add_table_cell(cell2, "sec_financial")
    
    print(f"   Index built: {len(index.nodes)} nodes")
    
    # 2. Mock encoders (using random embeddings for demo)
    print("\n2. Simulating Cross-Modal Encoding...")
    # Create dummy embeddings for each node
    for node_id, node in index.nodes.items():
        node.embedding = np.random.randn(768).astype(np.float32)
        node.embedding = node.embedding / np.linalg.norm(node.embedding)
    
    # 3. Mock Intent Classifier
    print("\n3. Simulating Intent Classification...")
    class DummyIntentClassifier:
        def predict_intent(self, emb):
            # Simple heuristic based on query keywords
            return "calculation"  # for demo
    
    # 4. Mock Fusion Network
    class DummyFusion:
        def gate(self, x):
            # Return random lambda between 0.2 and 0.8
            return torch.tensor([[0.5]])
    
    dummy_intent = DummyIntentClassifier()
    dummy_fusion = DummyFusion()
    
    # 5. Retriever
    print("\n4. Hierarchical Retrieval...")
    # Create dummy encoder wrapper
    class DummyEncoder:
        def encode_query(self, q):
            return np.random.randn(768).astype(np.float32)
    
    retriever = HierarchicalRetriever(
        index=index,
        encoder=DummyEncoder(),
        fusion_network=dummy_fusion,
        intent_classifier=dummy_intent,
        config={"l1_document_k": 2, "l2_section_k": 3, "l3_semantic_k": 5}
    )
    
    query = "What was the revenue growth?"
    evidence, lam, intent = retriever.retrieve(query)
    print(f"   Query: {query}")
    print(f"   Retrieved {len(evidence)} evidence nodes")
    print(f"   Fusion weight λ: {lam:.3f}")
    print(f"   Predicted intent: {intent}")
    
    # 6. Context building
    print("\n5. Building Context...")
    from src.retriever import ContextBuilder
    builder = ContextBuilder()
    context = builder.build_context(evidence)
    print(f"   Context length: {len(context)} chars")
    print(f"   Context preview: {context[:200]}...")
    
    print("\nDemo completed successfully!")


def demo_full_pipeline_if_models_exist():
    """Run full pipeline only if models are available"""
    try:
        from src.hierarchical_index import HierarchicalIndex
        from src.encoders import TextEncoder, TableEncoder, RetrievalEncoder
        from src.fusion import IntentClassifier, AdaptiveFusionNetwork
        from src.retriever import HierarchicalRetriever
        from src.generator import ResponseGenerator
        
        print("\n" + "=" * 60)
        print("Full Pipeline Demo (with actual models)")
        print("=" * 60)
        
        # Load models (requires pretrained weights)
        # This will fail if models not available
        text_enc = TextEncoder("ProsusAI/finbert")
        table_enc = TableEncoder("microsoft/tapex-large-finetuned-wtq")
        retrieval_enc = RetrievalEncoder(text_enc, table_enc)
        
        print("Models loaded successfully.")
        
    except Exception as e:
        print(f"\nFull pipeline not available: {e}")
        print("Please train or download models first.")


if __name__ == "__main__":
    demo_basic_components()
    demo_full_pipeline_if_models_exist()