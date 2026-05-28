"""
Evaluation metrics for financial QA.
Supports Multi-Doc-2025 five-subset design with extended slice metrics.
"""

import re
import numpy as np
from typing import List, Dict, Any
from collections import Counter, defaultdict

NUMERIC_REL_TOLERANCE = 1e-3


class QAEvaluator:
    """Question answering evaluation metrics."""

    @staticmethod
    def exact_match(prediction: str, ground_truth: str) -> bool:
        # Numerical GT: check if prediction contains a matching number.
        gt_nums = QAEvaluator._extract_numbers(ground_truth)
        if len(gt_nums) == 1 and re.fullmatch(r'\s*-?\d+(?:[.,]\d+)?%?\s*', ground_truth.strip()):
            pred_nums = QAEvaluator._extract_numbers(prediction)
            if pred_nums:
                gt = gt_nums[0]
                return any(abs(p - gt) / (abs(gt) + 1e-9) < NUMERIC_REL_TOLERANCE
                           for p in pred_nums)
            return False
        return (QAEvaluator._normalize(prediction) ==
                QAEvaluator._normalize(ground_truth))

    @staticmethod
    def f1_score(prediction: str, ground_truth: str) -> float:
        # Numerical GT: use proximity score instead of token F1
        gt_nums = QAEvaluator._extract_numbers(ground_truth)
        if len(gt_nums) == 1 and re.fullmatch(r'\s*-?\d+(?:[.,]\d+)?%?\s*', ground_truth.strip()):
            pred_nums = QAEvaluator._extract_numbers(prediction)
            if pred_nums:
                gt = gt_nums[0]
                best = min(abs(p - gt) / (abs(gt) + 1e-9) for p in pred_nums)
                return max(0.0, 1.0 - best)
            return 0.0

        pred_tokens = QAEvaluator._normalize(prediction).split()
        gt_tokens   = QAEvaluator._normalize(ground_truth).split()
        if not pred_tokens and not gt_tokens:
            return 1.0
        if not pred_tokens or not gt_tokens:
            return 0.0
        common  = Counter(pred_tokens) & Counter(gt_tokens)
        n_same  = sum(common.values())
        if n_same == 0:
            return 0.0
        precision = n_same / len(pred_tokens)
        recall    = n_same / len(gt_tokens)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def execution_accuracy(prediction: str, ground_truth: str) -> bool:
        pred_nums = QAEvaluator._extract_numbers(prediction)
        gt_nums   = QAEvaluator._extract_numbers(ground_truth)
        if not pred_nums and not gt_nums:
            return QAEvaluator.exact_match(prediction, ground_truth)
        if not gt_nums:
            return False
        # GT has 1 number: check if any predicted number is within tolerance.
        if len(gt_nums) == 1:
            gt = gt_nums[0]
            return any(abs(p - gt) / (abs(gt) + 1e-9) < NUMERIC_REL_TOLERANCE
                       for p in pred_nums)
        # GT has multiple numbers: require same count and pairwise match
        if len(pred_nums) != len(gt_nums):
            return False
        return all(abs(p - g) / (abs(g) + 1e-9) < NUMERIC_REL_TOLERANCE
                   for p, g in zip(pred_nums, gt_nums))

    @staticmethod
    def hallucination_rate(generated_answer: str, evidence: str) -> float:
        """
        Estimate hallucination rate for financial QA.
        Strategy: extract all numbers from the answer; a number is "supported"
        if it (or a value within tolerance) appears in the evidence.
        Returns the fraction of answer-numbers NOT supported by evidence.
        Falls back to 0.0 if the answer contains no numbers (text-only answers
        are hard to verify without NLI and we don't penalise them here).
        """
        answer_nums = QAEvaluator._extract_numbers(generated_answer)
        if not answer_nums:
            return 0.0

        evidence_nums = set(QAEvaluator._extract_numbers(evidence))
        unsupported = 0
        for a in answer_nums:
            # Supported if any evidence number is within tolerance.
            if not any(abs(a - e) / (abs(e) + 1e-9) < NUMERIC_REL_TOLERANCE
                       for e in evidence_nums):
                unsupported += 1
        return unsupported / len(answer_nums)

    @staticmethod
    def _normalize(s: str) -> str:
        s = s.lower()
        # Preserve decimal points in numbers before stripping punctuation
        s = re.sub(r'(\d)\.(\d)', r'\1_DOT_\2', s)
        s = re.sub(r'[^\w\s]', '', s)
        s = s.replace('_DOT_', '.')
        s = re.sub(r'\b(the|a|an|of|to|in|for|on|with|by|at)\b', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    @staticmethod
    def _extract_numbers(s: str) -> List[float]:
        result = []
        # Remove commas in numbers like 1,234,567 → 1234567
        s_clean = re.sub(r'(\d),(\d)', r'\1\2', s)
        # Handle units: $94M → 94000000, $1.2B → 1200000000, $500K → 500000
        s_clean = re.sub(r'\$\s*(-?\d+(?:\.\d+)?)\s*[Bb](?:illion)?',
                         lambda m: str(float(m.group(1)) * 1e9), s_clean)
        s_clean = re.sub(r'\$\s*(-?\d+(?:\.\d+)?)\s*[Mm](?:illion)?',
                         lambda m: str(float(m.group(1)) * 1e6), s_clean)
        s_clean = re.sub(r'\$\s*(-?\d+(?:\.\d+)?)\s*[Kk](?:illion)?',
                         lambda m: str(float(m.group(1)) * 1e3), s_clean)
        # Handle "X million/billion/thousand" text form
        s_clean = re.sub(r'(-?\d+(?:\.\d+)?)\s+billion',
                         lambda m: str(float(m.group(1)) * 1e9), s_clean, flags=re.I)
        s_clean = re.sub(r'(-?\d+(?:\.\d+)?)\s+million',
                         lambda m: str(float(m.group(1)) * 1e6), s_clean, flags=re.I)
        s_clean = re.sub(r'(-?\d+(?:\.\d+)?)\s+thousand',
                         lambda m: str(float(m.group(1)) * 1e3), s_clean, flags=re.I)
        for num in re.findall(r'-?\d+(?:\.\d+)?%?', s_clean):
            is_pct = num.endswith('%')
            try:
                v = float(num.rstrip('%'))
                result.append(v / 100 if is_pct else v)
            except ValueError:
                pass
        return result

    @staticmethod
    def _extract_claims(s: str) -> List[str]:
        return [c.strip() for c in re.split(r'[.!?]+', s) if len(c.strip()) > 10]


class BenchmarkEvaluator:
    """
    Evaluate on financial QA benchmarks.

    Slice metrics computed:
      - by intent  : calculation / trend / fact / comparison
      - by structure: cross_doc, cross_year, hybrid_modal
      - by subset  : S1 / S2 / S3 / S4 / S5  (Multi-Doc-2025)
      - by difficulty: L1 / L2 / L3
    """

    # Map IntentType.name variants → canonical short names used in GT
    _INTENT_NORM = {
        "calculation":    "calculation",
        "trend":          "trend",
        "trend_analysis": "trend",
        "fact":           "fact",
        "fact_finding":   "fact",
        "comparison":     "comparison",
    }

    @staticmethod
    def _norm_intent(intent: str) -> str:
        return BenchmarkEvaluator._INTENT_NORM.get(intent.lower(), intent.lower())

    def __init__(self):
        self.evaluator = QAEvaluator()

    def evaluate_dataset(self,
                         predictions:   List[Dict[str, Any]],
                         ground_truths: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Args:
            predictions  : list of dicts with keys 'answer', 'evidence', 'intent'
            ground_truths: list of dicts with keys 'answer', 'intent',
                           'execution_required', and optionally:
                           'is_cross_doc', 'is_cross_year', 'is_hybrid_modal',
                           'subset' (S1-S5), 'difficulty' (L1-L3)
        """
        base = {"em": [], "f1": [], "exec_acc": [], "hall_rate": [], "faithful_acc": []}

        intent_buckets = defaultdict(lambda: {"em": [], "f1": [], "exec_acc": []})
        struct_buckets = {
            "cross_doc":    [],
            "cross_year":   [],
            "hybrid_modal": [],
            # E.2 dual-axis: per-structure F1 slices
            "single_doc":   [],
            "cross_company":[],
        }
        subset_buckets     = defaultdict(list)   # S1-S5 -> f1 list
        difficulty_buckets = defaultdict(list)   # L1-L3 -> f1 list
        sector_buckets     = defaultdict(list)   # sector -> f1 list

        for pred, gt in zip(predictions, ground_truths):
            pred_ans = pred.get("answer", "")
            gt_ans   = gt.get("answer", "")
            intent   = self._norm_intent(gt.get("intent", "fact"))
            evidence = pred.get("evidence", "")

            em = self.evaluator.exact_match(pred_ans, gt_ans)
            f1 = self.evaluator.f1_score(pred_ans, gt_ans)

            base["em"].append(em)
            base["f1"].append(f1)

            if gt.get("execution_required", False):
                base["exec_acc"].append(
                    self.evaluator.execution_accuracy(pred_ans, gt_ans))

            # hall_rate: computed only when evidence is available
            hr = self.evaluator.hallucination_rate(pred_ans, evidence) if evidence else 0.0
            if evidence:
                base["hall_rate"].append(hr)

            # faithful_acc: #(correct answer AND supported by evidence) / #all_samples
            # "supported" = hall_rate < 0.5 (less than half of answer numbers are unsupported)
            # "correct"   = em == 1
            is_supported = (hr < 0.5)
            base["faithful_acc"].append(1.0 if (em and is_supported) else 0.0)

            # Intent slices
            intent_buckets[intent]["em"].append(em)
            intent_buckets[intent]["f1"].append(f1)
            if gt.get("execution_required") and intent == "calculation":
                intent_buckets[intent]["exec_acc"].append(
                    self.evaluator.execution_accuracy(pred_ans, gt_ans))

            # Structural slices
            is_cross_doc  = gt.get("is_cross_doc",    False)
            is_cross_year = gt.get("is_cross_year",   False)
            is_hybrid     = gt.get("is_hybrid_modal", False)
            # cross_company: S4/S5 samples have multiple companies
            companies = gt.get("companies", [])
            is_cross_company = len(companies) > 1 if isinstance(companies, list) else False

            if is_cross_doc:   struct_buckets["cross_doc"].append(f1)
            if is_cross_year:  struct_buckets["cross_year"].append(f1)
            if is_hybrid:      struct_buckets["hybrid_modal"].append(f1)
            if is_cross_company: struct_buckets["cross_company"].append(f1)
            if not is_cross_doc and not is_cross_year:
                struct_buckets["single_doc"].append(f1)

            # Subset slices (S1-S5)
            subset = gt.get("subset", "")
            if subset:
                subset_buckets[subset].append(f1)

            # Difficulty slices (L1-L4)
            diff = gt.get("difficulty", "")
            if diff:
                difficulty_buckets[diff].append(f1)

            # Sector slices (OOD generalization support)
            sector = gt.get("sector", "")
            if sector:
                sector_buckets[sector].append(f1)

        # Aggregate
        results = {
            "em":           np.mean(base["em"])           * 100,
            "f1":           np.mean(base["f1"])           * 100,
            "exec_acc":     np.mean(base["exec_acc"])     * 100 if base["exec_acc"]  else 0.0,
            "hall_rate":    np.mean(base["hall_rate"])    * 100 if base["hall_rate"] else 0.0,
            # faithful_acc denominator = all samples (not just those with evidence)
            "faithful_acc": np.mean(base["faithful_acc"]) * 100 if base["faithful_acc"] else 0.0,
        }

        for intent, im in intent_buckets.items():
            if im["em"]:
                results[f"{intent}_em"] = np.mean(im["em"]) * 100
            if im["f1"]:
                results[f"{intent}_f1"] = np.mean(im["f1"]) * 100
            if im["exec_acc"]:
                results[f"{intent}_exec_acc"] = np.mean(im["exec_acc"]) * 100

        for key, vals in struct_buckets.items():
            if vals:
                results[f"{key}_f1"] = np.mean(vals) * 100

        for subset, vals in subset_buckets.items():
            if vals:
                results[f"subset_{subset}_f1"] = np.mean(vals) * 100

        for diff, vals in difficulty_buckets.items():
            if vals:
                results[f"difficulty_{diff}_f1"] = np.mean(vals) * 100

        for sector, vals in sector_buckets.items():
            if vals:
                safe = sector.replace(" ", "_").replace("/", "_")
                results[f"sector_{safe}_f1"] = np.mean(vals) * 100

        return results
