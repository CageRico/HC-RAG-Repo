"""
Download 10-K HTML filings from SEC EDGAR for a list of companies.

Files are saved as  data/raw/{TICKER}_{YEAR}.html

Usage:
  # Default list of companies (matches the paper's benchmark companies)
  python scripts/download_sec_filings.py --output_dir ./data/raw

  # Custom tickers and years
  python scripts/download_sec_filings.py --tickers AAPL MSFT --years 2022 2023 2024
"""

import os
import sys
import time
import argparse
import requests
from typing import List, Optional

# SEC EDGAR requires a User-Agent header identifying the requester
HEADERS = {
    "User-Agent": "HC-RAG-Research research@example.com",
    "Accept-Encoding": "gzip, deflate",
}

# All 88 S&P 500 representative companies used in Multi-Doc-2025 (11 sectors × 8 companies)
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
DEFAULT_TICKERS = [t for tickers in SP500_COMPANIES.values() for t in tickers]
DEFAULT_YEARS = ["2024"]


def get_cik(ticker: str) -> Optional[str]:
    """Look up CIK number for a ticker via SEC EDGAR company search."""
    url = f"https://efts.sec.gov/LATEST/search-index?q=%22{ticker}%22&dateRange=custom&startdt=2020-01-01&forms=10-K"
    # Use the company tickers JSON endpoint instead — more reliable
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry.get("ticker", "").upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)
    return None


def get_10k_filing_url(cik: str, year: str) -> Optional[str]:
    """
    Find the 10-K filing for a given CIK whose fiscal year ends in `year`.
    10-K filings are typically submitted 2-4 months after fiscal year end,
    so we search filings from `year` and the first half of `year+1`.
    Returns the URL of the primary HTML document.
    """
    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(submissions_url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    # Collect all filing pages (recent + older pages)
    all_filings_pages = [data.get("filings", {}).get("recent", {})]
    for extra in data.get("filings", {}).get("files", []):
        extra_url = f"https://data.sec.gov/submissions/{extra['name']}"
        try:
            r2 = requests.get(extra_url, headers=HEADERS, timeout=30)
            r2.raise_for_status()
            all_filings_pages.append(r2.json())
            time.sleep(0.15)
        except Exception:
            pass

    next_year = str(int(year) + 1)
    for filings in all_filings_pages:
        forms = filings.get("form", [])
        dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])
        primary_docs = filings.get("primaryDocument", [])

        for form, date, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
            if form != "10-K":
                continue
            # Match filings submitted in `year` OR in Jan-Jun of `year+1`
            # (covers fiscal years ending Dec 31 of `year`)
            filed_in_year = date.startswith(year)
            filed_early_next = date.startswith(next_year) and date[5:7] <= "06"
            if filed_in_year or filed_early_next:
                accession_clean = accession.replace("-", "")
                doc_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession_clean}/{primary_doc}"
                )
                return doc_url

    return None


def download_filing(url: str, save_path: str) -> bool:
    """Download a filing HTML and save to disk."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=60)
        resp.raise_for_status()
        with open(save_path, "w", encoding="utf-8", errors="replace") as f:
            f.write(resp.text)
        return True
    except Exception as e:
        print(f"    [ERROR] {e}")
        return False


def download_all(tickers: List[str], years: List[str], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tickers:
        print(f"\n{ticker}")
        cik = get_cik(ticker)
        if not cik:
            print(f"  CIK not found, skipping.")
            continue
        print(f"  CIK: {cik}")

        for year in years:
            save_path = os.path.join(output_dir, f"{ticker}_{year}.html")
            if os.path.exists(save_path):
                print(f"  {year}: already exists, skipping.")
                continue

            filing_url = get_10k_filing_url(cik, year)
            if not filing_url:
                print(f"  {year}: no 10-K found.")
                continue

            print(f"  {year}: downloading {filing_url} ...")
            ok = download_filing(filing_url, save_path)
            if ok:
                size_kb = os.path.getsize(save_path) // 1024
                print(f"  {year}: saved ({size_kb} KB)")

            # Be polite to SEC servers — max 10 requests/second
            time.sleep(0.15)


def main():
    parser = argparse.ArgumentParser(description="Download SEC 10-K filings")
    parser.add_argument("--output_dir", default="./data/raw")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--years", nargs="+", default=DEFAULT_YEARS)
    args = parser.parse_args()

    print(f"Tickers : {args.tickers}")
    print(f"Years   : {args.years}")
    print(f"Output  : {args.output_dir}")

    download_all(args.tickers, args.years, args.output_dir)
    print("\nDownload complete.")


if __name__ == "__main__":
    main()
