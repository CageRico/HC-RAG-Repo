"""
Helpers for writing reproducibility-focused run metadata.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


def _checkpoint_status(config: Dict[str, Any]) -> Dict[str, Any]:
    checkpoint_dir = config.get("paths", {}).get("checkpoint_dir", "./checkpoints")
    expected = {
        "alignment": "align_checkpoint_best.pt",
        "intent": "intent_best.pt",
        "fusion": "fusion_best.pt",
    }
    resolved = {}
    for key, filename in expected.items():
        path = os.path.join(checkpoint_dir, filename)
        resolved[key] = {
            "filename": filename,
            "path": path,
            "exists": os.path.exists(path),
        }
    return resolved


def _normalize_scalar_or_list(value: Any) -> Any:
    if isinstance(value, (list, tuple, set)):
        items = [v for v in value if v is not None]
        if not items:
            return None
        unique = sorted(set(items))
        return unique[0] if len(unique) == 1 else unique
    return value


def build_run_metadata(
    *,
    config: Dict[str, Any],
    config_path: str,
    script_name: str,
    run_type: str,
    dataset: str,
    split: str,
    method: str,
    output_dir: str,
    retrieval_top_k: Any = None,
    final_evidence_budget: Any = None,
    max_samples: Optional[int] = None,
    workers: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metadata = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "run_type": run_type,
        "script": script_name,
        "config_path": config_path,
        "dataset": dataset,
        "split": split,
        "method": method,
        "output_dir": output_dir,
        "generator_model": config.get("models", {}).get("generator", ""),
        "text_encoder": config.get("models", {}).get("text_encoder", ""),
        "table_encoder": config.get("models", {}).get("table_encoder", ""),
        "local_files_only": config.get("models", {}).get("local_files_only", True),
        "openai_base_url": config.get("models", {}).get("openai_base_url", ""),
        "generation": {
            "max_tokens": config.get("generation", {}).get("max_tokens"),
            "temperature": config.get("generation", {}).get("temperature"),
            "top_p": config.get("generation", {}).get("top_p"),
        },
        "index": {
            "l1_document_k": config.get("index", {}).get("l1_document_k"),
            "l2_section_k": config.get("index", {}).get("l2_section_k"),
            "l3_semantic_k": config.get("index", {}).get("l3_semantic_k"),
            "chunk_size": config.get("index", {}).get("chunk_size"),
            "chunk_overlap": config.get("index", {}).get("chunk_overlap"),
        },
        "retrieval_top_k": _normalize_scalar_or_list(retrieval_top_k),
        "final_evidence_budget": _normalize_scalar_or_list(final_evidence_budget),
        "max_samples": max_samples,
        "workers": workers,
        "checkpoints": _checkpoint_status(config),
    }
    if extra:
        metadata.update(extra)
    return metadata


def save_run_metadata(metadata: Dict[str, Any],
                      output_dir: str,
                      run_prefix: str,
                      timestamp: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{run_prefix}_run_metadata_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    latest_path = os.path.join(output_dir, "run_metadata.json")
    with open(latest_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return path
