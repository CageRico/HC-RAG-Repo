"""
Small smoke test for baseline implementations.

The test runs a few samples through selected baselines and checks that:
1. answer() returns a non-empty string.
2. retrieve() returns non-empty chunks with content/doc_id/section/rank fields.
3. the answer does not start with an explicit LLM error marker.

Usage:
  python scripts/smoke_test_baselines.py
  python scripts/smoke_test_baselines.py --max_samples 5 --baselines bm25 vanilla_rag
"""

import argparse
import os
import sys
import traceback

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.prepare_datasets import load_dataset_split
from scripts.run_baselines import BASELINES


REQUIRED_RETRIEVE_KEYS = {"content", "doc_id", "section", "rank"}


def check_retrieve(result):
    if not result:
        return "retrieve() returned an empty list"
    missing = []
    for i, row in enumerate(result[:3]):
        for key in REQUIRED_RETRIEVE_KEYS:
            if key not in row:
                missing.append(f"chunk[{i}] missing field '{key}'")
    return "; ".join(missing) if missing else None


def check_answer(answer):
    if not answer or not isinstance(answer, str):
        return "answer() returned an empty or non-string value"
    if answer.startswith("[ERROR]"):
        return f"LLM call failed: {answer}"
    return None


def run_smoke(config, samples, baseline_names):
    results = {}
    for name in baseline_names:
        if name not in BASELINES:
            print(f"  [SKIP] {name} is not registered in BASELINES")
            continue

        print(f"\n{'=' * 50}")
        print(f"Testing baseline: {name}")
        errors = []
        try:
            baseline = BASELINES[name](config)
        except Exception as exc:
            print(f"  [FAIL] initialization failed: {exc}")
            results[name] = ["initialization failed: " + str(exc)]
            continue

        for i, sample in enumerate(samples):
            print(f"  Sample {i + 1}: {sample['question'][:60]}...")

            try:
                retrieved = baseline.retrieve(sample, top_k=5)
                err = check_retrieve(retrieved)
                if err:
                    errors.append(f"sample {i + 1} retrieve: {err}")
                else:
                    print(f"    retrieve OK ({len(retrieved)} chunks, "
                          f"top doc_id={retrieved[0].get('doc_id', '?')!r})")
            except Exception:
                tb = traceback.format_exc().strip().splitlines()[-1]
                errors.append(f"sample {i + 1} retrieve exception: {tb}")
                print(f"    retrieve FAIL: {tb}")

            try:
                answer = baseline.answer(sample)
                err = check_answer(answer)
                if err:
                    errors.append(f"sample {i + 1} answer: {err}")
                else:
                    print(f"    answer OK -> {answer[:80]!r}")
            except Exception:
                tb = traceback.format_exc().strip().splitlines()[-1]
                errors.append(f"sample {i + 1} answer exception: {tb}")
                print(f"    answer FAIL: {tb}")

        results[name] = errors

    print(f"\n{'=' * 50}")
    print("Smoke test summary")
    print(f"{'=' * 50}")
    all_pass = True
    for name, errors in results.items():
        if errors:
            all_pass = False
            print(f"  FAIL  {name}:")
            for error in errors:
                print(f"        - {error}")
        else:
            print(f"  PASS  {name}")

    print()
    if all_pass:
        print("All checks passed [OK]")
    else:
        print("Some checks failed; inspect the errors above.")
    return all_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--dataset", default="finqa")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument(
        "--baselines",
        nargs="+",
        default=["bm25", "vanilla_rag", "self_rag", "graphrag", "raptor", "tat_llm"],
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print(f"Loading dataset: {args.dataset}/{args.split} (first {args.max_samples} samples)")
    samples = load_dataset_split(args.dataset, args.split)[:args.max_samples]
    print(f"Samples: {len(samples)}")

    run_smoke(config, samples, args.baselines)


if __name__ == "__main__":
    main()
