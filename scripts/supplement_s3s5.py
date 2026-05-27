"""
Supplement S3 and S5 raw QA files for Multi-Doc-2025.

S3 creates additional cross-year questions from multiple 10-K sections.
S5 creates additional cross-company and cross-year question pairs.
Generated files use section/year suffixes and do not overwrite the original files.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.build_multidoc2025 import (
    SP500_COMPANIES,
    TICKER_TO_SECTOR,
    _S5_PROMPT,
    _call_llm,
    _load_section,
)


_S3_PROMPT_SECTION = """\
Company: {company}
Years available: {years}

--- FY{year_a} CONTENT ({section_title}) ---
{content_a}

--- FY{year_b} CONTENT ({section_title}) ---
{content_b}

--- FY{year_c} CONTENT ({section_title}) ---
{content_c}
--- END CONTENT ---

Subset: S3 (Cross-Year Trend)
Generate {n} question-answer pairs that require comparing data ACROSS the three fiscal years.
Focus on trends, changes, and evolution visible in {section_title}.
Questions must be answerable ONLY by reading all three years together.

Return a JSON array with keys:
  "question", "answer",
  "intent"           : "trend" or "calculation",
  "is_hybrid_modal"  : true if answer needs both text and table evidence,
  "is_cross_year"    : true,
  "years_required"   : ["{year_a}", "{year_b}", "{year_c}"],
  "evidence_section" : "{section_title}",
  "difficulty"       : "L2",
  "subset"           : "S3",
  "requires_calculation"

Return JSON array only, no extra text.
"""


BAD_KEYWORDS_Q = [
    "page", "excerpt", "provided excerpt", "provided content",
    "based on the provided", "in the provided", "the document",
    "this filing", "this report", "the text does not",
    "not mentioned", "not provided", "cannot determine",
    "does not specify", "no information",
]
BAD_KEYWORDS_A = [
    "not provided", "not mentioned", "cannot determine",
    "no information", "not specified",
]
VALID_INTENTS = {"calculation", "trend", "fact", "comparison"}


def _filter(pairs: List[Dict]) -> List[Dict]:
    filtered = []
    for pair in pairs:
        question = pair.get("question", "")
        answer = pair.get("answer", "")
        if len(question) < 20 or len(answer) < 3:
            continue
        if any(keyword in question.lower() for keyword in BAD_KEYWORDS_Q):
            continue
        if any(keyword in answer.lower() for keyword in BAD_KEYWORDS_A):
            continue
        if pair.get("intent") not in VALID_INTENTS:
            continue
        filtered.append(pair)
    return filtered


def supplement_s3(filings_dir: str, raw_dir: str, client, model: str,
                  n_per_section: int = 5, workers: int = 6):
    """Generate additional S3 samples for companies with at least three years of filings."""

    ticker_years: Dict[str, List[str]] = {}
    for fname in os.listdir(filings_dir):
        if not fname.endswith(".html"):
            continue
        parts = fname.replace(".html", "").split("_")
        ticker = parts[0].upper()
        year = parts[1] if len(parts) > 1 else "2024"
        ticker_years.setdefault(ticker, []).append(year)

    three_year = {ticker: sorted(years)
                  for ticker, years in ticker_years.items()
                  if len(years) >= 3}
    print(f"Companies with three-year data: {len(three_year)}")

    lock = threading.Lock()
    sections = [
        ("1A", "Item 1A. Risk Factors"),
        ("8", "Item 8. Financial Statements"),
        ("7", "Item 7. MD&A"),
    ]

    def _process(ticker: str, years: List[str]):
        year_a, year_b, year_c = years[0], years[1], years[2]
        sector = TICKER_TO_SECTOR.get(ticker, "Unknown")
        generated = 0

        for item_num, section_title in sections:
            save_path = os.path.join(raw_dir, f"S3_{ticker}_{item_num}.json")
            if os.path.exists(save_path):
                with lock:
                    print(f"  {ticker} {item_num}: exists, skipping")
                continue

            content_a = _load_section(filings_dir, ticker, year_a, [item_num], max_chars=1500)
            content_b = _load_section(filings_dir, ticker, year_b, [item_num], max_chars=1500)
            content_c = _load_section(filings_dir, ticker, year_c, [item_num], max_chars=1500)
            if not (content_a and content_b and content_c):
                continue

            pairs = _call_llm(client, _S3_PROMPT_SECTION.format(
                company=ticker,
                years=f"{year_a}, {year_b}, {year_c}",
                year_a=year_a,
                year_b=year_b,
                year_c=year_c,
                content_a=content_a,
                content_b=content_b,
                content_c=content_c,
                section_title=section_title,
                n=n_per_section,
            ), model=model)

            pairs = _filter(pairs)
            for pair in pairs:
                pair.update({
                    "company": ticker,
                    "sector": sector,
                    "is_cross_year": True,
                    "is_cross_doc": False,
                    "companies": [ticker],
                    "years_required": [year_a, year_b, year_c],
                    "year": year_c,
                    "subset": "S3",
                    "difficulty": "L2",
                })
                if pair.get("intent") in {"calculation", "trend", "comparison"}:
                    pair["is_hybrid_modal"] = True

            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(pairs, f, ensure_ascii=False, indent=2)
            generated += len(pairs)

        with lock:
            print(f"  {ticker}: added {generated} S3 samples")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, ticker, years): ticker
                   for ticker, years in three_year.items()}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"  [ERROR] S3 {futures[future]}: {exc}")


def supplement_s5(filings_dir: str, raw_dir: str, client, model: str,
                  n_per_pair: int = 8, workers: int = 4):
    """Generate additional S5 samples from cross-company pairs with overlapping years."""

    ticker_years: Dict[str, List[str]] = {}
    for fname in os.listdir(filings_dir):
        if not fname.endswith(".html"):
            continue
        parts = fname.replace(".html", "").split("_")
        ticker = parts[0].upper()
        year = parts[1] if len(parts) > 1 else "2024"
        ticker_years.setdefault(ticker, []).append(year)

    multi_year = {ticker: sorted(years)
                  for ticker, years in ticker_years.items()
                  if len(years) >= 2}

    tasks = []
    for sector, tickers in SP500_COMPANIES.items():
        available = [ticker for ticker in tickers if ticker in multi_year]
        if len(available) < 2:
            continue
        for i in range(len(available)):
            for j in range(i + 1, len(available)):
                ticker_a, ticker_b = available[i], available[j]
                year_a = multi_year[ticker_a][-2]
                year_b = multi_year[ticker_a][-1]
                if year_a in multi_year.get(ticker_b, []) and year_b in multi_year.get(ticker_b, []):
                    tasks.append((sector, ticker_a, ticker_b, year_a, year_b))

    print(f"Available S5 pairs: {len(tasks)}")
    lock = threading.Lock()

    def _process(sector: str, ticker_a: str, ticker_b: str, year_a: str, year_b: str):
        save_path = os.path.join(raw_dir, f"S5_{ticker_a}_{ticker_b}_{year_a}_{year_b}.json")
        if os.path.exists(save_path):
            with lock:
                print(f"  {ticker_a}+{ticker_b}: exists, skipping")
            return

        content_a1 = _load_section(filings_dir, ticker_a, year_a, ["7", "8"], max_chars=1000)
        content_a2 = _load_section(filings_dir, ticker_a, year_b, ["7", "8"], max_chars=1000)
        content_b1 = _load_section(filings_dir, ticker_b, year_a, ["7", "8"], max_chars=1000)
        content_b2 = _load_section(filings_dir, ticker_b, year_b, ["7", "8"], max_chars=1000)
        if not (content_a1 and content_a2 and content_b1 and content_b2):
            return

        pairs = _call_llm(client, _S5_PROMPT.format(
            sector=sector,
            company_a=ticker_a,
            company_b=ticker_b,
            year_a=year_a,
            year_b=year_b,
            content_a1=content_a1,
            content_a2=content_a2,
            content_b1=content_b1,
            content_b2=content_b2,
            n=n_per_pair,
        ), model=model)

        pairs = _filter(pairs)
        for pair in pairs:
            pair.update({
                "company": f"{ticker_a}+{ticker_b}",
                "sector": sector,
                "is_cross_doc": True,
                "is_cross_year": True,
                "is_hybrid_modal": True,
                "companies": [ticker_a, ticker_b],
                "years_required": [year_a, year_b],
                "year": year_b,
                "subset": "S5",
                "difficulty": "L3",
            })

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        with lock:
            print(f"  {ticker_a}+{ticker_b} ({year_a}-{year_b}): {len(pairs)} S5 samples")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_process, *task): task for task in tasks}
        for future in as_completed(futures):
            exc = future.exception()
            if exc:
                print(f"  [ERROR] S5 {futures[future]}: {exc}")


def merge_new_to_progress(raw_dir: str, progress_path: str):
    """Merge newly generated S3/S5 files into reviewed/_progress.json."""

    with open(progress_path, encoding="utf-8") as f:
        existing = json.load(f)

    existing_questions = {item.get("question", "").strip().lower()
                          for item in existing}
    new_pairs = []

    for fname in os.listdir(raw_dir):
        if not fname.endswith(".json"):
            continue

        parts = fname.replace(".json", "").split("_")
        if fname.startswith("S3_") and len(parts) == 3:
            is_new = True
        elif fname.startswith("S5_") and len(parts) == 5:
            is_new = True
        else:
            is_new = False

        if not is_new:
            continue

        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            pairs = json.load(f)

        for pair in pairs:
            question = pair.get("question", "").strip().lower()
            if question and question not in existing_questions:
                pair["_reviewed"] = True
                pair["_id"] = f"new_{len(existing) + len(new_pairs)}"
                new_pairs.append(pair)
                existing_questions.add(question)

    print(f"Added {len(new_pairs)} deduplicated samples")
    all_data = existing + new_pairs

    import shutil
    shutil.copy(progress_path, progress_path + ".bak2")
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)

    from collections import Counter
    subsets = Counter(item.get("subset") for item in all_data)
    print(f"Merged subset counts: {dict(sorted(subsets.items()))}")
    print(f"Merged total: {len(all_data)}")
    return len(new_pairs)


def main():
    import argparse
    import yaml
    from openai import OpenAI

    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["generate", "merge", "all"])
    parser.add_argument("--filings_dir", default="./data/raw")
    parser.add_argument("--output_dir", default="./data/multidoc2025")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--n_s3", type=int, default=5, help="Number of S3 samples per section")
    parser.add_argument("--n_s5", type=int, default=8, help="Number of S5 samples per pair")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    raw_dir = os.path.join(args.output_dir, "raw_qa")
    progress_path = os.path.join(args.output_dir, "reviewed", "_progress.json")

    if args.step in ("generate", "all"):
        cfg = {}
        if os.path.exists(args.config):
            with open(args.config, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        models_cfg = cfg.get("models", {})
        api_key = models_cfg.get("openai_api_key") or os.environ.get("OPENAI_API_KEY", "")
        base_url = models_cfg.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
        model = os.environ.get("QA_MODEL") or models_cfg.get("generator", "deepseek-v4-flash")
        client = OpenAI(api_key=api_key, base_url=base_url if base_url else None)
        print(f"Model: {model}  base_url: {base_url or 'default'}")

        print("\n=== Supplement S3 ===")
        supplement_s3(args.filings_dir, raw_dir, client, model,
                      n_per_section=args.n_s3, workers=args.workers)

        print("\n=== Supplement S5 ===")
        supplement_s5(args.filings_dir, raw_dir, client, model,
                      n_per_pair=args.n_s5, workers=args.workers)

    if args.step in ("merge", "all"):
        print("\n=== Merge new data into _progress.json ===")
        merge_new_to_progress(raw_dir, progress_path)


if __name__ == "__main__":
    main()
