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
from src.encoders import TextEncoder, TableEncoder, extract_alignment_projection_state
from src.fusion import (
    IntentClassifier,
    AdaptiveFusionNetwork,
    IntentType,
    compute_weak_lambda_target,
)
from src.evaluation import QAEvaluator
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

    def test_sequential_section_neighbors(self):
        doc = DocumentNode("doc1", "TestCorp", "2024", "Tech")
        self.index.add_document(doc)
        sec1 = SectionNode("sec1", "Item 1", 1, 10, 20)
        sec2 = SectionNode("sec2", "Item 1A", 1, 30, 40)
        sec3 = SectionNode("sec3", "Item 7", 1, 50, 60)
        self.index.add_section(sec1, "doc1")
        self.index.add_section(sec2, "doc1")
        self.index.add_section(sec3, "doc1")
        self.index.add_sequential_edge("sec1", "sec2")
        self.index.add_sequential_edge("sec2", "sec3")

        self.assertEqual(self.index.get_next_section("sec1").node_id, "sec2")
        self.assertEqual(self.index.get_previous_section("sec3").node_id, "sec2")
        self.assertEqual(
            [node.node_id for node in self.index.get_sequential_neighbors("sec2", hops=1)],
            ["sec1", "sec3"],
        )


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


class TestPaperAlignmentHelpers(unittest.TestCase):
    def test_extract_alignment_projection_state(self):
        checkpoint = {
            "model_state_dict": {
                "module.text_proj.0.weight": torch.randn(4, 4),
                "module.text_proj.0.bias": torch.randn(4),
                "module.text_proj.1.weight": torch.randn(4),
                "module.text_proj.1.bias": torch.randn(4),
                "module.table_proj.0.weight": torch.randn(4, 4),
                "module.table_proj.0.bias": torch.randn(4),
                "module.table_proj.1.weight": torch.randn(4),
                "module.table_proj.1.bias": torch.randn(4),
            }
        }
        text_state, table_state = extract_alignment_projection_state(checkpoint)
        self.assertIn("0.weight", text_state)
        self.assertIn("1.bias", text_state)
        self.assertIn("0.weight", table_state)
        self.assertIn("1.bias", table_state)

    def test_compute_weak_lambda_target(self):
        calc = compute_weak_lambda_target("calculation", is_hybrid_modal=False, subset="S2")
        trend = compute_weak_lambda_target("trend", is_hybrid_modal=False, subset="S3")
        hybrid_calc = compute_weak_lambda_target("calculation", is_hybrid_modal=True, subset="S5")
        fact = compute_weak_lambda_target("fact", is_hybrid_modal=False, subset="S1")

        self.assertLess(calc, 0.2)
        self.assertGreater(trend, 0.7)
        self.assertGreater(hybrid_calc, calc)
        self.assertGreater(fact, hybrid_calc)


class TestEvaluationTolerance(unittest.TestCase):
    def test_exact_match_uses_paper_tolerance(self):
        self.assertTrue(QAEvaluator.exact_match("100.05", "100"))
        self.assertFalse(QAEvaluator.exact_match("100.2", "100"))

    def test_execution_accuracy_uses_paper_tolerance(self):
        self.assertTrue(QAEvaluator.execution_accuracy("Revenue was 200.1", "200"))
        self.assertFalse(QAEvaluator.execution_accuracy("Revenue was 200.3", "200"))


if __name__ == "__main__":
    unittest.main()
