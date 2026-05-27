"""
Query-Aware Adaptive Fusion Network
Four-class intent taxonomy: calculation, trend analysis, fact finding, comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from enum import Enum


class IntentType(Enum):
    CALCULATION = 0      # numerical calculation questions
    TREND_ANALYSIS = 1   # trend analysis questions
    FACT_FINDING = 2     # fact lookup questions
    COMPARISON = 3       # cross-entity comparison questions


class IntentClassifier(nn.Module):
    """
    Multi-layer perceptron for query intent classification
    Four output classes: calculation, trend, fact, comparison
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256,
                 num_classes: int = 4, dropout: float = 0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, query_embedding: torch.Tensor) -> torch.Tensor:
        """Returns logits for intent classes"""
        return self.classifier(query_embedding)
    
    def predict_intent(self, query_embedding: torch.Tensor) -> IntentType:
        """Predict intent class from embedding"""
        with torch.no_grad():
            logits = self.forward(query_embedding)
            pred_class = torch.argmax(logits, dim=-1).item()
        return IntentType(pred_class)
    
    def get_intent_probs(self, query_embedding: torch.Tensor) -> np.ndarray:
        """Get probability distribution over intent classes"""
        with torch.no_grad():
            logits = self.forward(query_embedding)
            probs = F.softmax(logits, dim=-1)
        return probs.cpu().numpy()[0]


class AdaptiveFusionNetwork(nn.Module):
    """
    Query-aware adaptive fusion with dynamic modality weighting
    Fusion weight λ controls text vs table importance
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 256,
                 num_intents: int = 4):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Gating weight generator
        # Takes query embedding and intent features to produce fusion weight λ
        self.gate = nn.Sequential(
            nn.Linear(embedding_dim + num_intents, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Optional: modality-specific transformation
        self.text_transform = nn.Linear(embedding_dim, embedding_dim)
        self.table_transform = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, query_embedding: torch.Tensor, 
                text_embedding: torch.Tensor,
                table_embedding: torch.Tensor,
                intent_probs: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """
        Adaptive fusion with query-aware weighting
        
        Args:
            query_embedding: [batch_size, dim]
            text_embedding: [batch_size, dim]  
            table_embedding: [batch_size, dim]
            intent_probs: [batch_size, num_intents]
        
        Returns:
            fused_embedding: [batch_size, dim]
            lambda_weight: fusion coefficient
        """
        # Transform modalities
        text_transformed = self.text_transform(text_embedding)
        table_transformed = self.table_transform(table_embedding)
        
        # Combine query and intent for gating
        gate_input = torch.cat([query_embedding, intent_probs], dim=-1)
        lambda_weight = self.gate(gate_input)  # [batch_size, 1]
        
        # Weighted fusion: fused = λ * text + (1-λ) * table
        fused = lambda_weight * text_transformed + (1 - lambda_weight) * table_transformed
        
        return fused, lambda_weight.mean().item()
    
    def get_fusion_weight(self, query_embedding: torch.Tensor, 
                          intent_probs: torch.Tensor) -> float:
        """Get fusion weight λ for a single query"""
        with torch.no_grad():
            gate_input = torch.cat([query_embedding.unsqueeze(0), intent_probs.unsqueeze(0)], dim=-1)
            lambda_weight = self.gate(gate_input)
        return lambda_weight.item()


class HybridRetrievalFusion:
    """
    Combines retrieval results from text and table modalities
    with query-aware fusion weighting
    """
    
    def __init__(self, fusion_network: AdaptiveFusionNetwork,
                 intent_classifier: IntentClassifier):
        self.fusion_network = fusion_network
        self.intent_classifier = intent_classifier
        
    def fuse_retrieval_results(self, query: str, query_embedding: np.ndarray,
                               text_results: List[Tuple[any, float]],
                               table_results: List[Tuple[any, float]],
                               text_embeddings: List[np.ndarray],
                               table_embeddings: List[np.ndarray]) -> Tuple[List[any], float]:
        """
        Fuse text and table retrieval results based on query intent
        
        Args:
            query: original query string
            query_embedding: query embedding
            text_results: list of (chunk, score) from text retrieval
            table_results: list of (cell, score) from table retrieval
            text_embeddings: embeddings for text results
            table_embeddings: embeddings for table results
        
        Returns:
            fused_results: merged and re-ranked list of evidence
            fusion_weight: λ value used for fusion
        """
        intent_probs = self.intent_classifier.get_intent_probs(torch.from_numpy(query_embedding).float())
        intent_tensor = torch.from_numpy(intent_probs).float().unsqueeze(0)
        query_tensor = torch.from_numpy(query_embedding).float().unsqueeze(0)
        
        # Get fusion weight
        with torch.no_grad():
            gate_input = torch.cat([query_tensor, intent_tensor], dim=-1)
            lambda_weight = self.fusion_network.gate(gate_input).item()
        
        # Weight retrieval scores
        # For text: weight = λ, for table: weight = 1-λ
        text_weight = lambda_weight
        table_weight = 1 - lambda_weight
        
        # Combine and re-rank
        fused_with_scores = []
        
        for i, (chunk, score) in enumerate(text_results):
            weighted_score = score * text_weight
            fused_with_scores.append((chunk, weighted_score, "text"))
        
        for i, (cell, score) in enumerate(table_results):
            weighted_score = score * table_weight
            fused_with_scores.append((cell, weighted_score, "table"))
        
        # Sort by weighted score
        fused_with_scores.sort(key=lambda x: x[1], reverse=True)
        
        return fused_with_scores, lambda_weight


class IntentAwarePromptBuilder:
    """
    Builds generation prompts with intent-specific instructions
    """
    
    def __init__(self):
        self.intent_prompts = {
            IntentType.CALCULATION: """
You are a financial analyst. Answer the following question based ONLY on the provided evidence.
This is a NUMERICAL CALCULATION question.

Requirements:
1. Extract the exact numerical values from the tables
2. Show all intermediate calculation steps
3. Indicate whether each number is a direct quote or a calculation result
4. Provide final answer with proper units

Question: {query}

Evidence:
{evidence}

Answer (with step-by-step calculation):
""",
            IntentType.TREND_ANALYSIS: """
You are a financial analyst. Answer the following question based ONLY on the provided evidence.
This is a TREND ANALYSIS question.

Requirements:
1. Identify the trend direction (increase/decrease/stable)
2. Cite specific numbers and periods from the evidence
3. Explain contributing factors mentioned in the text
4. Provide a concise trend summary

Question: {query}

Evidence:
{evidence}

Analysis:
""",
            IntentType.FACT_FINDING: """
You are a financial analyst. Answer the following question based ONLY on the provided evidence.
This is a FACT-FINDING question.

Requirements:
1. Provide the exact factual answer from the evidence
2. Include source citation (document name, section)
3. If the answer is not found, state "Not found in provided documents"

Question: {query}

Evidence:
{evidence}

Answer:
""",
            IntentType.COMPARISON: """
You are a financial analyst. Answer the following question based ONLY on the provided evidence.
This is a CROSS-ENTITY COMPARISON question.

Requirements:
1. Identify the entities being compared (companies, years, or metrics)
2. Extract the relevant values for each entity from the evidence
3. Compute the absolute and/or percentage difference where applicable
4. State which entity leads and by how much
5. Briefly explain any factors mentioned in the evidence that drive the difference

Question: {query}

Evidence:
{evidence}

Comparison:
"""
        }
    
    def build_prompt(self, query: str, evidence: str, intent: IntentType) -> str:
        """Build prompt for the given intent type"""
        prompt_template = self.intent_prompts.get(intent, self.intent_prompts[IntentType.FACT_FINDING])
        return prompt_template.format(query=query, evidence=evidence)