"""
Load and explore the Multi-Doc-2025 dataset.
"""

import json
from collections import Counter
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent


def load_split(split: str):
    path = DATA_DIR / f"{split}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def print_stats(data, split_name):
    print(f"
=== {split_name}: {len(data)} samples ===")
    print("Intents:   ", dict(Counter(s["intent"] for s in data)))
    print("Subsets:   ", dict(Counter(s["subset"] for s in data)))
    print("Difficulty:", dict(Counter(s["difficulty"] for s in data)))
    cross_doc  = sum(1 for s in data if s["is_cross_doc"])
    cross_year = sum(1 for s in data if s["is_cross_year"])
    hybrid     = sum(1 for s in data if s["is_hybrid_modal"])
    print(f"cross_doc={cross_doc}  cross_year={cross_year}  hybrid_modal={hybrid}")


def main():
    train = load_split("train")
    val   = load_split("val")
    test  = load_split("test")

    print_stats(train, "train")
    print_stats(val,   "val")
    print_stats(test,  "test")

    print("
=== Example sample (test[0]) ===")
    print(json.dumps(test[0], indent=2))

    print("
=== Filter examples ===")
    s5_test    = [x for x in test if x["subset"] == "S5"]
    calc_test  = [x for x in test if x["intent"] == "calculation"]
    cross_test = [x for x in test if x["is_cross_doc"]]
    print(f"S5 (Full-Cross) test samples:      {len(s5_test)}")
    print(f"Calculation intent test samples:   {len(calc_test)}")
    print(f"Cross-document test samples:       {len(cross_test)}")


if __name__ == "__main__":
    main()
