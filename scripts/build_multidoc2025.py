"""
Build the Multi-Doc-2025 dataset.

Five subsets (S1-S5) covering orthogonal reasoning dimensions:
  S1  Single-Doc-Fact    single company, single year, fact/calculation   L1  ~400
  S2  Single-Doc-Table   single company, single year, pure table         L1  ~300
  S3  Cross-Year         same company, 2022->2024 trend                  L2  ~400
  S4  Cross-Company      same sector, two companies, FY2024 comparison   L2  ~500
  S5  Full-Cross         cross-company + cross-year + hybrid-modal       L3  ~400

Total target: 2000 QA pairs, 89 companies, 11 sectors, 3 years (2022-2024)

Intent taxonomy (4 classes):
  calculation  numerical computation from tables/text
  trend        temporal change across years
  fact         factual retrieval from text
  comparison   side-by-side comparison of two entities

Usage:
  python scripts/build_multidoc2025.py generate --filings_dir ./data/raw
  python scripts/build_multidoc2025.py review   --output_dir  ./data/multidoc2025
  python scripts/build_multidoc2025.py split    --output_dir  ./data/multidoc2025
"""

import os
import sys
import json
import re
import random
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# S&P 500 sector mapping (11 sectors x 8 companies)
# ---------------------------------------------------------------------------
SP500_COMPANIES = {
    "Information Technology": ["AAPL", "MSFT", "NVDA", "AVGO", "ORCL", "CRM", "AMD", "INTC"],
    "Financials":             ["JPM",  "BAC",  "WFC",  "GS",   "MS",   "BLK", "AXP", "C"],
    "Healthcare":             ["JNJ",  "UNH",  "LLY",  "PFE",  "ABBV", "MRK", "TMO", "ABT"],
    "Consumer Discretionary": ["AMZN", "TSLA", "HD",   "MCD",  "NKE",  "LOW", "SBUX","TGT"],
    "Consumer Staples":       ["WMT",  "PG",   "KO",   "PEP",  "COST", "PM",  "MO",  "CL"],
    "Industrials":            ["GE",   "HON",  "UPS",  "CAT",  "DE",   "LMT", "RTX", "BA"],
    "Communication Services": ["META", "GOOGL","NFLX", "DIS",  "CMCSA","T",   "VZ",  "CHTR"],
    "Energy":                 ["XOM",  "CVX",  "COP",  "SLB",  "EOG",  "MPC", "PSX", "VLO"],
    "Materials":              ["LIN",  "APD",  "ECL",  "DD",   "NEM",  "FCX", "ALB", "IFF"],
    "Real Estate":            ["AMT",  "PLD",  "CCI",  "EQIX", "PSA",  "DLR", "O",   "WELL"],
    "Utilities":              ["NEE",  "DUK",  "SO",   "D",    "AEP",  "EXC", "XEL", "ED"],
}

TICKER_TO_SECTOR = {t: s for s, tickers in SP500_COMPANIES.items() for t in tickers}

# ---------------------------------------------------------------------------
# LLM prompt templates per subset
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = (
    "You are a senior financial analyst creating a rigorous QA benchmark from SEC 10-K filings. "
    "Generate precise, verifiable question-answer pairs. Answers must be directly supported by "
    "the provided content. Quote numerical values exactly as they appear in the source."
)

_S1_PROMPT = """\
Company: {company}  Fiscal Year: {year}  Section: {section_title}
Subset: S1 (Single-Document Fact/Calculation)

--- CONTENT ---
{content}
--- END CONTENT ---

Generate {n} question-answer pairs covering FACT retrieval and CALCULATION tasks.
Each pair must be answerable from this section alone.

Return a JSON array. Each element must have exactly these keys:
  "question"         : clear, specific question string
  "answer"           : concise answer (quote numbers exactly)
  "intent"           : "fact" or "calculation"
  "is_hybrid_modal"  : true if answer requires BOTH text and table evidence, else false
  "evidence_section" : section title where answer is found
  "difficulty"       : "L1"
  "subset"           : "S1"
  "requires_calculation": true if numerical computation is needed, else false

Return JSON array only, no extra text.
"""

_S2_PROMPT = """\
Company: {company}  Fiscal Year: {year}  Section: {section_title}
Subset: S2 (Single-Document Table Reasoning)

--- CONTENT ---
{content}
--- END CONTENT ---

Generate {n} question-answer pairs that require reading and reasoning over TABLES only.
Questions should involve multi-row/multi-column lookups, ratio calculations, or
comparisons within a single table.

Return a JSON array with keys:
  "question", "answer",
  "intent"           : must be "calculation",
  "is_hybrid_modal"  : must be false (table only),
  "evidence_section", "difficulty": "L1", "subset": "S2",
  "requires_calculation": true

Return JSON array only.
"""

_S3_PROMPT = """\
Company: {company}
Years available: {years}

--- FY{year_a} CONTENT (Item 7 / MD&A) ---
{content_a}

--- FY{year_b} CONTENT (Item 7 / MD&A) ---
{content_b}

--- FY{year_c} CONTENT (Item 7 / MD&A) ---
{content_c}
--- END CONTENT ---

Subset: S3 (Cross-Year Trend)
Generate {n} question-answer pairs that require comparing data ACROSS the three fiscal years.
Focus on: revenue/profit trends, margin evolution, strategic shifts, risk factor changes.

Return a JSON array with keys:
  "question", "answer",
  "intent"           : must be "trend",
  "is_hybrid_modal", "is_cross_year": true,
  "years_required"   : list of years needed (e.g. ["2022","2023","2024"]),
  "evidence_section", "difficulty": "L2", "subset": "S3",
  "requires_calculation"

Return JSON array only.
"""

_S4_PROMPT = """\
Sector: {sector}
Company A: {company_a}  FY{year}
Company B: {company_b}  FY{year}

--- {company_a} CONTENT ---
{content_a}

--- {company_b} CONTENT ---
{content_b}
--- END CONTENT ---

Subset: S4 (Cross-Company Comparison)
Generate {n} question-answer pairs that require information from BOTH companies.
Focus on: competitive positioning, relative financial performance, market share,
strategic differences within the same sector.

Return a JSON array with keys:
  "question", "answer",
  "intent"           : must be "comparison",
  "is_hybrid_modal", "is_cross_doc": true, "is_cross_year": false,
  "companies"        : ["{company_a}", "{company_b}"],
  "evidence_section", "difficulty": "L2", "subset": "S4",
  "requires_calculation"

Return JSON array only.
"""

_S5_PROMPT = """\
Sector: {sector}
{company_a} vs {company_b}, years {year_a} and {year_b}.

{company_a} {year_a}: {content_a1}
{company_a} {year_b}: {content_a2}
{company_b} {year_a}: {content_b1}
{company_b} {year_b}: {content_b2}

Generate {n} hard QA pair requiring both companies, both years, and numerical evidence.
Return JSON array with keys: question, answer, intent ("comparison" or "trend"),
is_hybrid_modal (true), is_cross_doc (true), is_cross_year (true),
companies (["{company_a}","{company_b}"], years_required (["{year_a}","{year_b}"]),
evidence_section, difficulty ("L3"), subset ("S5"), requires_calculation.
JSON only, no extra text.
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_extract_fn():
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "hcrag_utils",
        pathlib.Path(__file__).parent.parent / "src" / "utils.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.extract_sections_from_10k

_extract_sections_from_10k = None


def _load_section(filings_dir: str, ticker: str, year: str,
                  item_nums: List[str], max_chars: int = 3000) -> str:
    global _extract_sections_from_10k
    if _extract_sections_from_10k is None:
        _extract_sections_from_10k = _get_extract_fn()
    path = os.path.join(filings_dir, f"{ticker}_{year}.html")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8", errors="replace") as f:
        html = f.read()
    sections = _extract_sections_from_10k(html, [])
    parts = []
    for sec in sections:
        if sec["item_num"] in item_nums:
            parts.append(sec["text"][:max_chars])
    return "\n\n".join(parts)


def _call_llm(client, prompt: str, model: str = "deepseek-v4-flash", max_tokens: int = 4096,
               max_retries: int = 5) -> List[Dict]:
    import time
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                temperature=0.7,
                max_tokens=max_tokens,
            )
            raw = resp.choices[0].message.content
            if raw is None:
                print(f"    [WARN] Model returned None content. finish_reason={resp.choices[0].finish_reason}")
                return []
            raw = raw.strip()
            if not raw:
                print(f"    [WARN] Model returned empty string. finish_reason={resp.choices[0].finish_reason}")
                return []
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            raw = raw.strip()
            return json.loads(raw)
        except json.JSONDecodeError as e:
            print(f"    [WARN] JSON parse failed: {e} | raw[:200]={raw[:200]!r}")
            return []
        except Exception as e:
            err_str = str(e)
            if "503" in err_str or "service_unavailable" in err_str or "too busy" in err_str.lower():
                wait = 10 * (attempt + 1)
                print(f"    [RETRY {attempt+1}/{max_retries}] Service busy, retrying after {wait}s...")
                time.sleep(wait)
                continue
            print(f"    [WARN] LLM call failed: {e}")
            return []
    print("    [WARN] Maximum retries reached; skipping.")
    return []


def _enrich(pairs: List[Dict], ticker: str, year: str, sector: str) -> List[Dict]:
    for p in pairs:
        p.setdefault("company",            ticker)
        p.setdefault("year",               year)
        p.setdefault("sector",             sector)
        p.setdefault("is_cross_doc",       False)
        p.setdefault("is_cross_year",      False)
        p.setdefault("is_hybrid_modal",    False)
        p.setdefault("companies",          [ticker])
        p.setdefault("years_required",     [year])
        p.setdefault("requires_calculation", False)
    return pairs


# ---------------------------------------------------------------------------
# Step 1: Generate candidates
# ---------------------------------------------------------------------------

def generate_candidates(filings_dir: str, output_dir: str, n_per_call: int = 3,
                        config_path: str = "config.yaml"):
    try:
        from openai import OpenAI
        import yaml
    except ImportError:
        print("openai/pyyaml not installed. Run: pip install openai pyyaml")
        sys.exit(1)

    # Load API config: config.yaml takes precedence, env vars as fallback
    cfg = {}
    if os.path.exists(config_path):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    models_cfg = cfg.get("models", {})

    api_key  = (models_cfg.get("openai_api_key")
                or os.environ.get("OPENAI_API_KEY", ""))
    base_url = (models_cfg.get("openai_base_url")
                or os.environ.get("OPENAI_BASE_URL"))
    model    = (os.environ.get("QA_MODEL")
                or models_cfg.get("generator", "deepseek-v4-flash"))

    if not api_key:
        print("ERROR: No API key found. Set openai_api_key in config.yaml or OPENAI_API_KEY env var.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key,
        base_url=base_url if base_url else None,
    )
    print(f"Using model: {model}  base_url: {base_url or 'default'}")
    raw_dir = os.path.join(output_dir, "raw_qa")
    os.makedirs(raw_dir, exist_ok=True)

    html_files = sorted(f for f in os.listdir(filings_dir) if f.endswith(".html"))

    # Build ticker -> years index
    ticker_years: Dict[str, List[str]] = {}
    for fname in html_files:
        parts  = fname.replace(".html", "").split("_")
        ticker = parts[0].upper()
        year   = parts[1] if len(parts) > 1 else "2024"
        ticker_years.setdefault(ticker, []).append(year)

    _print_lock = threading.Lock()

    def _log(*args):
        with _print_lock:
            print(*args)

    # ---- S1 + S2 ----
    print("\n=== S1/S2: Single-document subsets ===")

    def _process_s1s2(fname):
        parts  = fname.replace(".html", "").split("_")
        ticker = parts[0].upper()
        year   = parts[1] if len(parts) > 1 else "2024"
        sector = TICKER_TO_SECTOR.get(ticker, "Unknown")
        save_path = os.path.join(raw_dir, f"S1S2_{ticker}_{year}.json")
        if os.path.exists(save_path):
            _log(f"  {ticker} {year}: already exists, skipping.")
            return
        pairs = []
        for item_num, title in [("1", "Business"), ("1A", "Risk Factors"),
                                  ("7", "MD&A"), ("7A", "Market Risk"),
                                  ("8", "Financial Statements")]:
            content = _load_section(filings_dir, ticker, year, [item_num], max_chars=2000)
            if not content:
                continue
            pairs.extend(_enrich(
                _call_llm(client, _S1_PROMPT.format(
                    company=ticker, year=year,
                    section_title=f"Item {item_num}. {title}",
                    content=content, n=n_per_call), model=model),
                ticker, year, sector))
            if item_num in ("7", "8"):
                pairs.extend(_enrich(
                    _call_llm(client, _S2_PROMPT.format(
                        company=ticker, year=year,
                        section_title=f"Item {item_num}. {title}",
                        content=content, n=n_per_call), model=model),
                    ticker, year, sector))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        _log(f"  {ticker} {year}: {len(pairs)} pairs (S1+S2)")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_s1s2, fname): fname for fname in html_files}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                _log(f"  [ERROR] {futures[fut]}: {exc}")

    # ---- S3: cross-year ----
    print("\n=== S3: Cross-year subset ===")
    three_year = {t: sorted(y) for t, y in ticker_years.items() if len(y) >= 3}

    def _process_s3(ticker, years):
        sector    = TICKER_TO_SECTOR.get(ticker, "Unknown")
        save_path = os.path.join(raw_dir, f"S3_{ticker}.json")
        if os.path.exists(save_path):
            _log(f"  {ticker}: already exists, skipping.")
            return
        ya, yb, yc = years[0], years[1], years[2]
        ca = _load_section(filings_dir, ticker, ya, ["7"], max_chars=1200)
        cb = _load_section(filings_dir, ticker, yb, ["7"], max_chars=1200)
        cc = _load_section(filings_dir, ticker, yc, ["7"], max_chars=1200)
        if not (ca and cb and cc):
            return
        pairs = _call_llm(client, _S3_PROMPT.format(
            company=ticker, years=f"{ya}, {yb}, {yc}",
            year_a=ya, year_b=yb, year_c=yc,
            content_a=ca, content_b=cb, content_c=cc,
            n=n_per_call * 2), model=model)
        for p in pairs:
            p.update({"company": ticker, "sector": sector,
                       "is_cross_year": True, "is_cross_doc": False,
                       "companies": [ticker], "years_required": [ya, yb, yc],
                       "subset": "S3", "difficulty": "L2"})
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        _log(f"  {ticker}: {len(pairs)} pairs (S3)")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_s3, t, y): t for t, y in three_year.items()}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                _log(f"  [ERROR] S3 {futures[fut]}: {exc}")

    # ---- S4: cross-company ----
    print("\n=== S4: Cross-company subset ===")
    s4_tasks = []
    for sector, tickers in SP500_COMPANIES.items():
        available = [t for t in tickers if "2024" in ticker_years.get(t, [])]
        if len(available) < 2:
            continue
        random.shuffle(available)
        for i in range(0, min(len(available) - 1, 6), 2):
            s4_tasks.append((sector, available[i], available[i + 1]))

    def _process_s4(sector, ta, tb):
        save_path = os.path.join(raw_dir, f"S4_{ta}_{tb}.json")
        if os.path.exists(save_path):
            _log(f"  {ta}+{tb}: already exists, skipping.")
            return
        ca = _load_section(filings_dir, ta, "2024", ["7", "8"], max_chars=2000)
        cb = _load_section(filings_dir, tb, "2024", ["7", "8"], max_chars=2000)
        if not (ca and cb):
            return
        pairs = _call_llm(client, _S4_PROMPT.format(
            sector=sector, company_a=ta, company_b=tb, year="2024",
            content_a=ca, content_b=cb, n=n_per_call * 2), model=model)
        for p in pairs:
            p.update({"company": f"{ta}+{tb}", "year": "2024", "sector": sector,
                       "is_cross_doc": True, "is_cross_year": False,
                       "companies": [ta, tb], "years_required": ["2024"],
                       "subset": "S4", "difficulty": "L2"})
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        _log(f"  {ta}+{tb}: {len(pairs)} pairs (S4)")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_s4, s, ta, tb): (ta, tb) for s, ta, tb in s4_tasks}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                _log(f"  [ERROR] S4 {futures[fut]}: {exc}")

    # ---- S5: full-cross ----
    print("\n=== S5: Full-cross subset ===")
    s5_tasks = []
    for sector, tickers in SP500_COMPANIES.items():
        multi = [t for t in tickers if len(ticker_years.get(t, [])) >= 2]
        if len(multi) < 2:
            continue
        random.shuffle(multi)
        for i in range(0, min(len(multi) - 1, 4), 2):
            ta, tb       = multi[i], multi[i + 1]
            common_years = sorted(set(ticker_years.get(ta, [])) & set(ticker_years.get(tb, [])))
            if len(common_years) < 2:
                continue
            s5_tasks.append((sector, ta, tb, common_years[0], common_years[-1]))

    def _process_s5(sector, ta, tb, ya, yb):
        save_path = os.path.join(raw_dir, f"S5_{ta}_{tb}.json")
        if os.path.exists(save_path):
            _log(f"  {ta}+{tb}: already exists, skipping.")
            return
        ca1 = _load_section(filings_dir, ta, ya, ["7"], max_chars=300)
        ca2 = _load_section(filings_dir, ta, yb, ["7"], max_chars=300)
        cb1 = _load_section(filings_dir, tb, ya, ["7"], max_chars=300)
        cb2 = _load_section(filings_dir, tb, yb, ["7"], max_chars=300)
        if not (ca1 and ca2 and cb1 and cb2):
            return
        # Call 3 times with n=1 to avoid long JSON output being truncated
        all_pairs = []
        for _ in range(3):
            result = _call_llm(client, _S5_PROMPT.format(
                sector=sector, company_a=ta, company_b=tb,
                years=f"{ya}, {yb}", year_a=ya, year_b=yb,
                content_a1=ca1, content_a2=ca2,
                content_b1=cb1, content_b2=cb2,
                n=1), model=model, max_tokens=4096)
            all_pairs.extend(result)
        pairs = all_pairs
        for p in pairs:
            p.update({"company": f"{ta}+{tb}", "sector": sector,
                       "is_cross_doc": True, "is_cross_year": True,
                       "is_hybrid_modal": True,
                       "companies": [ta, tb], "years_required": [ya, yb],
                       "subset": "S5", "difficulty": "L3"})
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(pairs, f, ensure_ascii=False, indent=2)
        _log(f"  {ta}+{tb}: {len(pairs)} pairs (S5)")

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_process_s5, s, ta, tb, ya, yb): (ta, tb)
                   for s, ta, tb, ya, yb in s5_tasks}
        for fut in as_completed(futures):
            exc = fut.exception()
            if exc:
                _log(f"  [ERROR] S5 {futures[fut]}: {exc}")

    print("\nGeneration complete.")


# ---------------------------------------------------------------------------
# Step 2: LLM-based automatic review
# ---------------------------------------------------------------------------

_REVIEW_PROMPT = ('Rate this financial QA pair. Q: {question} A: {answer} '
                  '(subset={subset} intent={intent} difficulty={difficulty} company={company} year={year}). '
                  'Reply with JSON only: {{"score":0-12,"verdict":"accept"or"reject",'
                  '"corrected_intent":"calculation/trend/fact/comparison",'
                  '"corrected_difficulty":"L1/L2/L3","reason":"one sentence"}}')


def review_candidates(output_dir: str, auto: bool = False, config_path: str = "config.yaml",
                      min_score: int = 7, workers: int = 16):
    raw_dir      = os.path.join(output_dir, "raw_qa")
    reviewed_dir = os.path.join(output_dir, "reviewed")
    os.makedirs(reviewed_dir, exist_ok=True)

    all_candidates = []
    for fname in sorted(os.listdir(raw_dir)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(raw_dir, fname), encoding="utf-8") as f:
            all_candidates.extend(json.load(f))

    progress_path = os.path.join(reviewed_dir, "_progress.json")
    reviewed: List[Dict] = []
    reviewed_ids: set = set()

    if os.path.exists(progress_path):
        with open(progress_path, encoding="utf-8") as f:
            reviewed = json.load(f)
        reviewed_ids = {r["_id"] for r in reviewed}
        print(f"Resuming: {len(reviewed)} already reviewed.")

    pending = [(i, c) for i, c in enumerate(all_candidates) if str(i) not in reviewed_ids]
    print(f"{len(pending)} candidates to review.")

    if auto:
        _run_auto_review(pending, reviewed, reviewed_ids, progress_path,
                         config_path, min_score, workers)
    else:
        _run_manual_review(pending, reviewed, progress_path)

    print(f"\nReview complete. {len(reviewed)} accepted pairs saved.")


def _run_auto_review(pending, reviewed, reviewed_ids, progress_path,
                     config_path, min_score, workers):
    """Rule-based quality filter; no LLM calls needed."""

    # Keywords indicating the model talked about document structure instead of content
    BAD_KEYWORDS = [
        "page", "excerpt", "provided excerpt", "provided content",
        "based on the provided", "in the provided", "the document",
        "this filing", "this report", "the text does not",
        "not mentioned", "not provided", "cannot determine",
        "does not specify", "no information",
    ]

    VALID_INTENTS     = {"calculation", "trend", "fact", "comparison"}
    VALID_DIFFICULTY  = {"L1", "L2", "L3"}
    VALID_SUBSETS     = {"S1", "S2", "S3", "S4", "S5"}

    accepted = rejected = 0

    for orig_idx, cand in pending:
        q = cand.get("question", "")
        a = cand.get("answer",   "")
        q_lower = q.lower()
        a_lower = a.lower()

        # Hard filters
        if len(q) < 20:                          # question too short
            rejected += 1; continue
        if len(a) < 3:                           # answer too short
            rejected += 1; continue
        if any(kw in q_lower for kw in BAD_KEYWORDS):  # meta-question
            rejected += 1; continue
        if any(kw in a_lower for kw in ["not provided", "not mentioned",
                                         "cannot determine", "no information",
                                         "not specified"]):
            rejected += 1; continue
        if cand.get("intent")     not in VALID_INTENTS:
            rejected += 1; continue
        if cand.get("difficulty") not in VALID_DIFFICULTY:
            rejected += 1; continue
        if cand.get("subset")     not in VALID_SUBSETS:
            rejected += 1; continue

        cand = dict(cand)
        cand["_reviewed"] = True
        cand["_id"]       = str(orig_idx)
        reviewed.append(cand)
        accepted += 1

    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump(reviewed, f, ensure_ascii=False, indent=2)
    print(f"Rule-based review done: accepted={accepted}, rejected={rejected}")


def _run_manual_review(pending, reviewed, progress_path):
    print(f"{len(pending)} candidates to review.")
    print("a=accept  e=edit answer  i=edit intent  r=reject  q=quit\n")

    for idx, (orig_idx, cand) in enumerate(pending):
        print(f"\n[{idx+1}/{len(pending)}] "
              f"Subset:{cand.get('subset','?')}  "
              f"Company:{cand.get('company')}  "
              f"Year:{cand.get('year','multi')}")
        print(f"  Intent   : {cand.get('intent')}  Difficulty:{cand.get('difficulty')}")
        print(f"  Q: {cand.get('question')}")
        print(f"  A: {cand.get('answer')}")

        action = input("  Action [a/e/i/r/q]: ").strip().lower()
        if action == "q":
            break
        elif action == "r":
            continue

        cand = dict(cand)
        if action == "e":
            cand["answer"] = input("  New answer: ").strip()
        elif action == "i":
            cand["intent"] = input(
                "  New intent [calculation/trend/fact/comparison]: ").strip()

        cand["_reviewed"] = True
        cand["_id"]       = str(orig_idx)
        reviewed.append(cand)

        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump(reviewed, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Step 3: Finalize splits (primary-company-disjoint, subset-balanced)
# ---------------------------------------------------------------------------

def finalize_splits(output_dir: str, target: int = 2500,
                    train_ratio: float = 0.70, val_ratio: float = 0.10,
                    s1s2_cap: int = 600):
    """
    s1s2_cap: max S1+S2 pairs to keep (per subset, so 600 each = 1200 total max).
              Set to 0 to disable capping.
    """
    reviewed_dir  = os.path.join(output_dir, "reviewed")
    progress_path = os.path.join(reviewed_dir, "_progress.json")

    if not os.path.exists(progress_path):
        print("No reviewed data found. Run the 'review' step first.")
        sys.exit(1)

    with open(progress_path, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total reviewed: {len(data)}")

    # --- Downsample S1 and S2 to balance subset distribution ---
    from collections import defaultdict
    by_subset = defaultdict(list)
    for item in data:
        by_subset[item.get("subset", "S1")].append(item)

    balanced = []
    for subset, items in by_subset.items():
        if subset in ("S1", "S2") and s1s2_cap > 0:
            random.shuffle(items)
            balanced.extend(items[:s1s2_cap])
        else:
            balanced.extend(items)
    data = balanced
    print(f"After S1/S2 cap ({s1s2_cap} each): {len(data)} pairs")

    # Print subset distribution after capping
    subset_counts = defaultdict(int)
    for item in data:
        subset_counts[item.get("subset", "?")] += 1
    print(f"Subset dist (pre-split): {dict(sorted(subset_counts.items()))}")

    for i, item in enumerate(data):
        item["id"] = f"md2025_{i:04d}"

    companies = list({item.get("company", "").split("+")[0] for item in data})
    random.shuffle(companies)
    n_train   = int(len(companies) * train_ratio)
    n_val     = int(len(companies) * val_ratio)
    train_cos = set(companies[:n_train])
    val_cos   = set(companies[n_train:n_train + n_val])

    train, val, test = [], [], []
    for item in data:
        co = item.get("company", "").split("+")[0]
        if co in train_cos:
            train.append(item)
        elif co in val_cos:
            val.append(item)
        else:
            test.append(item)

    # Use all data; no further sampling truncation.
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(output_dir, f"{split_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        print(f"  {split_name}: {len(split_data)} samples -> {path}")

    from collections import Counter
    all_split = train + val + test
    n = len(all_split)
    print(f"\nTotal: {n}")
    print(f"Intent dist : {dict(Counter(x.get('intent') for x in all_split))}")
    print(f"Subset dist : {dict(Counter(x.get('subset') for x in all_split))}")
    print(f"Difficulty  : {dict(Counter(x.get('difficulty') for x in all_split))}")
    cross_doc  = sum(1 for x in all_split if x.get("is_cross_doc"))
    cross_year = sum(1 for x in all_split if x.get("is_cross_year"))
    hybrid     = sum(1 for x in all_split if x.get("is_hybrid_modal"))
    print(f"Cross-doc   : {cross_doc}  ({100*cross_doc/n:.1f}%)")
    print(f"Cross-year  : {cross_year} ({100*cross_year/n:.1f}%)")
    print(f"Hybrid-modal: {hybrid}     ({100*hybrid/n:.1f}%)")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build Multi-Doc-2025 dataset")
    parser.add_argument("step", choices=["generate", "review", "split"])
    parser.add_argument("--filings_dir", default="./data/raw")
    parser.add_argument("--output_dir",  default="./data/multidoc2025")
    parser.add_argument("--n_per_call",  type=int, default=3)
    parser.add_argument("--target",      type=int, default=2500)
    parser.add_argument("--s1s2_cap",    type=int, default=600,
                        help="Max pairs per S1/S2 subset (0=no cap)")
    parser.add_argument("--auto",        action="store_true",
                        help="Use LLM for automatic review (review step only)")
    parser.add_argument("--min_score",   type=int, default=7,
                        help="Minimum LLM review score to accept (0-12, default 7)")
    parser.add_argument("--workers",     type=int, default=16,
                        help="Concurrent workers for auto-review")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.step == "generate":
        generate_candidates(args.filings_dir, args.output_dir, args.n_per_call)
    elif args.step == "review":
        review_candidates(args.output_dir, auto=args.auto,
                          min_score=args.min_score, workers=args.workers)
    elif args.step == "split":
        finalize_splits(args.output_dir, target=args.target, s1s2_cap=args.s1s2_cap)


if __name__ == "__main__":
    main()
