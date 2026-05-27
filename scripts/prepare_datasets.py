"""
Download and convert benchmark datasets from HuggingFace to HC-RAG format.

Output format per sample:
  ground_truth: {"question": str, "answer": str, "intent": str, "execution_required": bool}

Run:
  python scripts/prepare_datasets.py --output_dir ./data/benchmarks
"""

import os
import sys
import json
import argparse
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Intent heuristic
# ---------------------------------------------------------------------------

_CALC_KEYWORDS = {
    "calculate", "compute", "what is the", "how much", "ratio", "margin",
    "percentage", "growth", "increase", "decrease", "total", "sum", "average",
    "per share", "rate", "return", "yield", "multiple",
}
_TREND_KEYWORDS = {
    "trend", "over time", "year over year", "yoy", "quarter over quarter",
    "changed", "evolved", "history", "compare", "versus", "vs",
    "improved", "declined", "grew", "fell",
}
_FACT_EXPLANATION_KEYWORDS = {
    "explain", "why", "reason", "cause", "driven by", "drove",
    "attributed to", "due to", "because", "factor", "describe", "elaborate",
    "what drove", "what caused",
}


def _infer_intent(question: str) -> str:
    q = question.lower()
    if any(k in q for k in _TREND_KEYWORDS):
        return "trend"
    if any(k in q for k in _FACT_EXPLANATION_KEYWORDS):
        return "fact"
    if any(k in q for k in _CALC_KEYWORDS):
        return "calculation"
    return "fact"


def _needs_execution(question: str) -> bool:
    q = question.lower()
    return any(k in q for k in _CALC_KEYWORDS)


def _split_flare_prompt(raw_question: str):
    """
    flare-* datasets pack everything into one string:
      'Please answer ... Context: <ctx>\nQuestion: <q>\nAnswer:'
    Returns (question_text, context_text).
    """
    q, ctx = raw_question, ""
    # Extract context
    for ctx_marker in ("Context:", "context:"):
        idx = raw_question.find(ctx_marker)
        if idx != -1:
            rest = raw_question[idx + len(ctx_marker):]
            # Context ends at the next "Question:" marker
            for q_marker in ("\nQuestion:", "\nquestion:", "\nQ:"):
                q_idx = rest.find(q_marker)
                if q_idx != -1:
                    ctx = rest[:q_idx].strip()
                    rest = rest[q_idx + len(q_marker):]
                    break
            else:
                ctx = rest.strip()
                rest = ""
            # Extract question from remainder
            for end in ("\nAnswer:", "\nA:", "\n"):
                e_idx = rest.find(end)
                if e_idx != -1:
                    rest = rest[:e_idx]
                    break
            q = rest.strip()
            break
    return q or raw_question, ctx


# ---------------------------------------------------------------------------
# FinQA  (ChanceFocus/flare-finqa — parquet, no loading script)
# ---------------------------------------------------------------------------

def convert_finqa(output_dir: str):
    from datasets import load_dataset
    print("Downloading FinQA ...")
    # ibm/finqa uses a deprecated loading script; ChanceFocus/flare-finqa is a
    # clean parquet mirror with the same content (query/answer/text columns).
    ds = load_dataset("ChanceFocus/flare-finqa")

    split_map = {"train": "train", "valid": "validation", "test": "test"}
    for src_split, dst_split in split_map.items():
        if src_split not in ds:
            continue
        samples = []
        for row in ds[src_split]:
            raw_q = row.get("query", "") or row.get("question", "")
            answer = row.get("answer", "")
            # flare-finqa: "text" field is the question; "query" contains the full
            # prompt with Context: ... \nQuestion: ... \nAnswer:
            question, context = _split_flare_prompt(raw_q)
            # If parsing failed, fall back to the "text" field as question
            # (do NOT use "text" as context — it is the question, not the document)
            if not question:
                question = row.get("text", "")
            if not question or not answer:
                continue
            samples.append({
                "question": question,
                "answer": str(answer),
                "intent": _infer_intent(question),
                "execution_required": _needs_execution(question),
                "context": context,
                "is_cross_doc": False,
                "is_hybrid_modal": False,
            })
        _save(samples, output_dir, "finqa", dst_split)


# ---------------------------------------------------------------------------
# TAT-QA  (ChanceFocus/flare-tatqa — parquet mirror)
# ---------------------------------------------------------------------------

def convert_tatqa(output_dir: str):
    from datasets import load_dataset
    print("Downloading TAT-QA ...")
    # ChanceFocus/flare-tatqa: the "query" field contains a prompt wrapper with
    # the full table+text context embedded after "Context:". The "text" field
    # is just the question itself (NOT the context). We must extract context
    # from the query field using _split_flare_prompt.
    ds = load_dataset("ChanceFocus/flare-tatqa")

    split_map = {"train": "train", "valid": "validation", "test": "test"}
    for src_split, dst_split in split_map.items():
        if src_split not in ds:
            continue
        samples = []
        for row in ds[src_split]:
            raw_q = row.get("query", "") or row.get("question", "")
            answer = row.get("answer", "")
            if isinstance(answer, list):
                answer = ", ".join(str(a) for a in answer)
            # flare-tatqa: same format as flare-finqa — "text" is the question,
            # "query" contains Context: ... \nQuestion: ... \nAnswer:
            question, context = _split_flare_prompt(raw_q)
            if not question:
                question = row.get("text", "")
            if not context:
                context = ""
            if not question or not answer:
                continue
            samples.append({
                "question": question,
                "answer": str(answer),
                "intent": _infer_intent(question),
                "execution_required": _needs_execution(question),
                "context": context,
                "is_cross_doc": False,
                "is_hybrid_modal": True,  # TAT-QA requires table+text by design
            })
        _save(samples, output_dir, "tatqa", dst_split)


# ---------------------------------------------------------------------------
# FinanceBench  (PatronusAI/financebench)
# ---------------------------------------------------------------------------

def convert_financebench(output_dir: str):
    from datasets import load_dataset
    print("Downloading FinanceBench ...")
    ds = load_dataset("PatronusAI/financebench", trust_remote_code=True)

    # FinanceBench only has a single split
    split_name = list(ds.keys())[0]
    samples = []
    for row in ds[split_name]:
        question = row.get("question", "")
        answer = row.get("answer", "")
        if not question or not answer:
            continue
        samples.append({
            "question": question,
            "answer": str(answer),
            "intent": _infer_intent(question),
            "execution_required": _needs_execution(question),
            "context": row.get("evidence", ""),
            "company": row.get("company", ""),
            "fiscal_year": str(row.get("fiscal_year", "")),
            "is_cross_doc": row.get("question_type", "") in ("multi-doc", "cross-doc"),
            "is_hybrid_modal": row.get("evidence_type", "") in ("table", "hybrid"),
        })
    _save(samples, output_dir, "financebench", "test")


# ---------------------------------------------------------------------------
# DocFinQA  (kensho/DocFinQA)
# ---------------------------------------------------------------------------

def convert_docfinqa(output_dir: str):
    from datasets import load_dataset
    print("Downloading DocFinQA ...")
    # Linq-AI-Research/DocFinQA no longer exists on Hub; kensho/DocFinQA is the
    # canonical version maintained by the original authors.
    ds = load_dataset("kensho/DocFinQA")

    split_map = {
        "train": "train", "validation": "validation", "test": "test",
        "dev": "validation",  # some versions use "dev"
    }
    found_any = False
    for src_split, dst_split in split_map.items():
        if src_split not in ds:
            continue
        found_any = True
        samples = []
        for row in ds[src_split]:
            # kensho/DocFinQA uses capitalized column names
            question = row.get("Question", "") or row.get("question", "") or row.get("query", "")
            answer = row.get("Answer", "") or row.get("answer", "")
            if not question or not answer:
                continue
            samples.append({
                "question": question,
                "answer": str(answer),
                "intent": _infer_intent(question),
                "execution_required": _needs_execution(question),
                "context": row.get("Context", "") or row.get("context", "") or row.get("text", ""),
                "is_cross_doc": False,
                "is_hybrid_modal": True,
            })
        _save(samples, output_dir, "docfinqa", dst_split)
    if not found_any:
        for src_split in ds.keys():
            samples = []
            for row in ds[src_split]:
                question = row.get("Question", "") or row.get("question", "") or row.get("query", "")
                answer = row.get("Answer", "") or row.get("answer", "")
                if not question or not answer:
                    continue
                samples.append({
                    "question": question,
                    "answer": str(answer),
                    "intent": _infer_intent(question),
                    "execution_required": _needs_execution(question),
                    "context": row.get("Context", "") or row.get("context", "") or row.get("text", ""),
                    "is_cross_doc": False,
                    "is_hybrid_modal": True,
                })
            _save(samples, output_dir, "docfinqa", src_split)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save(samples: List[Dict], output_dir: str, name: str, split: str):
    os.makedirs(os.path.join(output_dir, name), exist_ok=True)
    path = os.path.join(output_dir, name, f"{split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(samples)} samples -> {path}")


def load_dataset_split(dataset: str, split: str, data_dir: str = "./data/benchmarks") -> List[Dict]:
    """Load a converted dataset split for use in evaluation.

    For Multi-Doc-2025, data_dir is ignored and the canonical path
    ./data/multidoc2025/{split}.json is used instead.
    """
    if dataset == "multidoc2025":
        path = os.path.join("./data/multidoc2025", f"{split}.json")
    else:
        path = os.path.join(data_dir, dataset, f"{split}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}. "
            f"Run prepare_datasets.py (public benchmarks) or "
            f"build_multidoc2025.py (Multi-Doc-2025) first."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and convert benchmark datasets")
    parser.add_argument("--output_dir", default="./data/benchmarks")
    parser.add_argument(
        "--datasets", nargs="+",
        default=["finqa", "tatqa", "financebench", "docfinqa"],
        choices=["finqa", "tatqa", "financebench", "docfinqa"],
        help="Multi-Doc-2025 is built separately via build_multidoc2025.py",
    )
    args = parser.parse_args()

    converters = {
        "finqa": convert_finqa,
        "tatqa": convert_tatqa,
        "financebench": convert_financebench,
        "docfinqa": convert_docfinqa,
    }

    for name in args.datasets:
        try:
            converters[name](args.output_dir)
        except Exception as e:
            print(f"[ERROR] {name}: {e}")

    print("\nAll done. Datasets saved to:", args.output_dir)


if __name__ == "__main__":
    main()
