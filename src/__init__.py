"""
HC-RAG: Hierarchical Cross-Modal Retrieval-Augmented Generation

Heavy dependencies (torch, transformers) are imported lazily to avoid
import errors when only lightweight modules (utils) are needed.
"""


def _lazy_imports():
    from .hierarchical_index import (HierarchicalIndex, DocumentNode, SectionNode,
                                      TextChunkNode, TableCellNode, NodeType, EdgeType)
    from .encoders import TextEncoder, TableEncoder, CrossModalAligner, RetrievalEncoder
    from .fusion import IntentClassifier, AdaptiveFusionNetwork, IntentType, IntentAwarePromptBuilder
    from .retriever import HierarchicalRetriever, ContextBuilder
    from .generator import ResponseGenerator, GenerationResult
    from .evaluation import QAEvaluator, BenchmarkEvaluator
    return locals()


__all__ = [
    "HierarchicalIndex", "DocumentNode", "SectionNode",
    "TextChunkNode", "TableCellNode", "NodeType", "EdgeType",
    "TextEncoder", "TableEncoder", "CrossModalAligner", "RetrievalEncoder",
    "IntentClassifier", "AdaptiveFusionNetwork", "IntentType", "IntentAwarePromptBuilder",
    "HierarchicalRetriever", "ContextBuilder",
    "ResponseGenerator", "GenerationResult",
    "QAEvaluator", "BenchmarkEvaluator",
]
