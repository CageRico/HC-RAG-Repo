"""
Unit tests for HC-RAG components
"""

import unittest
import sys
import os
import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.hierarchical_index import (
    HierarchicalIndex, DocumentNode, SectionNode, TextChunkNode, TableCellNode, NodeType
)
from src.encoders import TextEncoder, TableEncoder
from src.fusion import IntentClassifier, AdaptiveFusionNetwork, IntentType
from src.utils import chunk_text, extract_tables_from_html, normalize_number


class TestHierarchicalIndex(unittest.TestCase):
    def setUp(self):
        self.index = HierarchicalIndex({})
        
    def test_add_document(self):
        doc = DocumentNode("doc1", "TestCorp", "2024", "Tech")
        self.index.add_document(doc)
        self.assertIn("doc1", self.index.doc_nodes)
        
    def test_add_section(self):
        doc = DocumentNode("doc1", "TestCorp", "2024", "Tech")
        self.index.add_document(doc)
        section = SectionNode("sec1", "Item 1", 1, 0, 100)
        self.index.add_section(section, "doc1")
        self.assertIn("sec1", self.index.section_nodes)
        children = self.index.get_children("doc1")
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].node_id, "sec1")
        
    def test_add_text_chunk(self):
        doc = DocumentNode("doc1", "TestCorp", "2024", "Tech")
        self.index.add_document(doc)
        section = SectionNode("sec1", "Item 1", 1, 0, 100)
        self.index.add_section(section, "doc1")
        chunk = TextChunkNode("chunk1", "Some text", 0, 10)
        self.index.add_text_chunk(chunk, "sec1")
        self.assertIn("chunk1", self.index.text_chunk_nodes)
        
    def test_cross_doc_edge(self):
        doc1 = DocumentNode("doc1", "TestCorp", "2024", "Tech")
        doc2 = DocumentNode("doc2", "OtherCorp", "2024", "Tech")
        self.index.add_document(doc1)
        self.index.add_document(doc2)
        self.index.add_cross_doc_edge("doc1", "doc2", "same_industry")
        related = self.index.get_related_documents("doc1")
        self.assertEqual(len(related), 1)
        self.assertEqual(related[0].node_id, "doc2")


class TestUtils(unittest.TestCase):
    def test_chunk_text(self):
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, chunk_size=100, overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertLessEqual(len(chunks[0]), 120)
        
    def test_extract_tables_from_html(self):
        html = """
        <table>
            <tr><th>Year</th><th>Revenue</th></tr>
            <tr><td>2023</td><td>$100M</td></tr>
            <tr><td>2024</td><td>$115M</td></tr>
        </table>
        """
        tables = extract_tables_from_html(html)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0]['header'], ['Year', 'Revenue'])
        self.assertEqual(len(tables[0]['rows']), 2)
        
    def test_normalize_number(self):
        self.assertEqual(normalize_number("$1.2M"), 1_200_000)
        self.assertEqual(normalize_number("15%"), 0.15)
        self.assertEqual(normalize_number("(2.3)"), -2.3)
        self.assertEqual(normalize_number("1,234"), 1234)


class TestIntentClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = IntentClassifier(input_dim=768, hidden_dim=256, num_classes=4)
        
    def test_forward(self):
        x = torch.randn(2, 768)
        logits = self.classifier(x)
        self.assertEqual(logits.shape, (2, 4))
        
    def test_predict_intent(self):
        # Mock embedding
        emb = torch.randn(1, 768)
        # The classifier is untrained; just ensure it runs
        intent = self.classifier.predict_intent(emb)
        self.assertIn(intent, list(IntentType))


class TestFusionNetwork(unittest.TestCase):
    def setUp(self):
        self.fusion = AdaptiveFusionNetwork(embedding_dim=768, hidden_dim=256, num_intents=4)
        
    def test_fusion(self):
        query = torch.randn(1, 768)
        text = torch.randn(1, 768)
        table = torch.randn(1, 768)
        intent = torch.randn(1, 4)
        fused, lam = self.fusion(query, text, table, intent)
        self.assertEqual(fused.shape, (1, 768))
        self.assertGreaterEqual(lam, 0)
        self.assertLessEqual(lam, 1)


if __name__ == "__main__":
    unittest.main()
