"""
Evaluate predictions on Multi-Doc-2025.

Usage:
    python evaluate.py --predictions predictions.json --split test

predictions.json format:
    [{"id": "md2025_0001", "prediction": "43.32%"}, ...]
"""

import json
import re
import argparse
from collections import Counter, defaultdict
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent


def normalize(s: str) -> str:
    s = s.lower()
    s = re.sub(r"(\d)\.(\d)", r"_DOT_", s)
    s = re.sub(r"[^\w\s]", "", s)
    s = s.replace("_DOT_", ".")
    s = re.sub(r"(the|a|an|of|to|in|for|on|with|by|at)", "", s)
    return re.sub(r"\s+", " ", s).strip()


def extract_numbers(s: str):
    s = re.sub(r"(\d),(\d)", r"", s)
    s = re.sub(r"$\s*(-?\d+(?:\.\d+)?)\s*[Bb]", lambda m: str(float(m.group(1))*1e9), s)
    s = re.sub(r"$\s*(-?\d+(?:\.\d+)?)\s*[Mm]", lambda m: str(float(m.group(1))*1e6), s)
    s = re.sub(r"$\s*(-?\d+(?:\.\d+)?)\s*[Kk]", lambda m: str(float(m.group(1))*1e3), s)
    nums = []
    for n in re.findall(r"-?\d+(?:\.\d+)?%?", s):
        try:
            v = float(n.rstrip("%"))
            nums.append(v / 100 if n.endswith("%") else v)
        except ValueError:
            pass
    return nums


def exact_match(pred: str, gt: str) -> bool:
    gt_nums = extract_numbers(gt)
    if len(gt_nums) == 1 and re.fullmatch(r"\s*-?\d+(?:[.,]\d+)?%?\s*", gt.strip()):
        pred_nums = extract_numbers(pred)
        return any(abs(p - gt_nums[0]) / (abs(gt_nums[0]) + 1e-9) < 0.01 for p in pred_nums) if pred_nums else False
    return normalize(pred) == normalize(gt)


def f1_score(pred: str, gt: str) -> float:
    gt_nums = extract_numbers(gt)
    if len(gt_nums) == 1 and re.fullmatch(r"\s*-?\d+(?:[.,]\d+)?%?\s*", gt.strip()):
        pred_nums = extract_numbers(pred)
        if pred_nums:
            best = min(abs(p - gt_nums[0]) / (abs(gt_nums[0]) + 1e-9) for p in pred_nums)
            return max(0.0, 1.0 - best)
        return 0.0
    pt = normalize(pred).split()
    gt_t = normalize(gt).split()
    if not pt and not gt_t:
        return 1.0
    if not pt or not gt_t:
        return 0.0
    common = sum((Counter(pt) & Counter(gt_t)).values())
    if common == 0:
        return 0.0
    p = common / len(pt)
    r = common / len(gt_t)
    return 2 * p * r / (p + r)


def evaluate(predictions, ground_truths):
    pred_map = {p["id"]: p["prediction"] for p in predictions}
    em_list, f1_list = [], []
    slice_f1 = defaultdict(list)

    for gt in ground_truths:
        gid = gt["id"]
        pred = pred_map.get(gid, "")
        em = exact_match(pred, gt["answer"])
        f1 = f1_score(pred, gt["answer"])
        em_list.append(em)
        f1_list.append(f1)
        slice_f1[f"intent_{gt['intent']}"].append(f1)
        slice_f1[f"subset_{gt['subset']}"].append(f1)
        slice_f1[f"difficulty_{gt['difficulty']}"].append(f1)
        if gt["is_cross_doc"]:   slice_f1["cross_doc"].append(f1)
        if gt["is_cross_year"]:  slice_f1["cross_year"].append(f1)
        if gt["is_hybrid_modal"]: slice_f1["hybrid_modal"].append(f1)

    results = {
        "em":  sum(em_list) / len(em_list) * 100,
        "f1":  sum(f1_list) / len(f1_list) * 100,
        "n":   len(em_list),
    }
    for k, v in slice_f1.items():
        results[k + "_f1"] = sum(v) / len(v) * 100
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="test", choices=["train","val","test"])
    args = parser.parse_args()

    with open(args.predictions, encoding="utf-8") as f:
        predictions = json.load(f)
    with open(DATA_DIR / f"{args.split}.json", encoding="utf-8") as f:
        ground_truths = json.load(f)

    results = evaluate(predictions, ground_truths)
    print(f"
Results on {args.split} ({results['n']} samples):")
    print(f"  EM:  {results['em']:.2f}%")
    print(f"  F1:  {results['f1']:.2f}%")
    print("
Slice metrics:")
    for k in sorted(k for k in results if k.endswith("_f1")):
        print(f"  {k:<30s} {results[k]:.2f}%")


if __name__ == "__main__":
    main()
