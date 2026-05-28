"""
Response generation layer.
Uses DeepSeek V4-flash or a configured local model with intent-aware prompts.
"""

import json
import openai
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import torch

from .fusion import IntentType, IntentAwarePromptBuilder
from .retriever import ContextBuilder
from .hierarchical_index import BaseNode


@dataclass
class GenerationResult:
    answer: str
    sources: List[Dict[str, str]]
    fusion_weight: float
    intent: IntentType
    confidence: float


class ResponseGenerator:
    """
    LLM-based response generator with source tracing.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model_name = config.get("model_name", "deepseek-v4-flash")
        self.max_tokens = config.get("max_tokens", 1024)
        self.temperature = config.get("temperature", 0.0)
        self.local_files_only = config.get("local_files_only", True)
        
        # Initialize components
        self.prompt_builder = IntentAwarePromptBuilder()
        self.context_builder = ContextBuilder()
        
        # Initialize OpenAI-compatible client (supports OpenAI, DeepSeek, etc.)
        import os
        self.use_openai = True
        base_url = config.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
        api_key  = config.get("openai_api_key",  "") or os.environ.get("OPENAI_API_KEY", "")
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url if base_url else None,
        )
    
    def _init_local_model(self):
        """Initialize local LLM (e.g., via transformers)"""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        self.local_tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=self.local_files_only,
        )
        self.local_model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16,
            device_map="auto",
            local_files_only=self.local_files_only,
        )
        self.local_model.eval()
    
    def generate(self, query: str, evidence_nodes: List[BaseNode],
                 fusion_weight: float, intent: IntentType) -> GenerationResult:
        """
        Generate answer from retrieved evidence
        
        Args:
            query: user question
            evidence_nodes: retrieved evidence from retriever
            fusion_weight: λ value used in fusion
            intent: predicted query intent
        
        Returns:
            GenerationResult with answer and sources
        """
        # Build context from evidence and enforce the released context budget.
        context = self.context_builder.build_context(evidence_nodes)
        words = context.split()
        if len(words) > 3000:
            context = " ".join(words[:3000])

        # HC-RAG uses the intent-aware prompt family defined in the paper.
        prompt = self.prompt_builder.build_prompt(query=query, evidence=context, intent=intent)
        
        # Generate response
        if self.use_openai:
            response = self._generate_openai(prompt)
        else:
            response = self._generate_local(prompt)
        
        # Extract sources from evidence
        sources = self._extract_sources(evidence_nodes)
        
        # Calculate confidence (simplified)
        confidence = self._estimate_confidence(response, evidence_nodes)
        
        return GenerationResult(
            answer=response,
            sources=sources,
            fusion_weight=fusion_weight,
            intent=intent,
            confidence=confidence
        )
    
    def _generate_openai(self, prompt: str) -> str:
        """Generate using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a financial analyst assistant. Answer questions based only on the provided evidence."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                extra_body={"thinking": {"type": "disabled"}}
            )
            content = response.choices[0].message.content
            return content if content else ""
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _generate_local(self, prompt: str) -> str:
        """Generate using local model"""
        inputs = self.local_tokenizer(prompt, return_tensors="pt").to(self.local_model.device)
        
        with torch.no_grad():
            outputs = self.local_model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                temperature=self.temperature,
                do_sample=self.temperature > 0
            )
        
        response = self.local_tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove prompt from response
        response = response[len(prompt):].strip()
        return response
    
    def _extract_sources(self, evidence_nodes: List[BaseNode]) -> List[Dict[str, str]]:
        """Extract source information from evidence nodes"""
        sources = []
        for node in evidence_nodes[:5]:  # Top 5 sources
            source = {
                "type": node.node_type.value,
                "content": str(node.content)
            }
            
            # Add metadata
            if "title" in node.metadata:
                source["title"] = node.metadata["title"]
            if "table_id" in node.metadata:
                source["table_id"] = node.metadata["table_id"]
            if "value" in node.metadata:
                source["value"] = node.metadata["value"]
            
            sources.append(source)
        
        return sources
    
    def _estimate_confidence(self, answer: str, evidence_nodes: List[BaseNode]) -> float:
        """
        Simple confidence estimation based on evidence relevance
        """
        if not evidence_nodes:
            return 0.1
        
        # Check if answer contains placeholders or error indicators
        if "not found" in answer.lower() or "unable to" in answer.lower():
            return 0.3
        
        # Check if answer seems substantial
        if len(answer) < 20:
            return 0.5
        
        # Base confidence
        return min(0.95, 0.7 + min(len(evidence_nodes), 10) * 0.02)
