---
name: Multi-Doc-2025
license: cc-by-4.0
task_categories:
  - question-answering
task_ids:
  - extractive-qa
  - open-domain-qa
language:
  - en
tags:
  - finance
  - sec-filings
  - 10-k
  - multi-document
  - cross-modal
  - table-qa
  - financial-qa
  - rag-benchmark
pretty_name: Multi-Doc-2025
size_categories:
  - 1K<n<10K
---


# Multi-Doc-2025

**Multi-Doc-2025** is a financial question-answering benchmark built from SEC 10-K annual reports of S&P 500 companies. It is the first benchmark to simultaneously evaluate **cross-company**, **cross-year**, and **hybrid-modal** (text + table) reasoning in a single unified framework.

---

## At a Glance

| Property | Value |
|----------|-------|
| Total QA pairs | **2,327** |
| Source documents | 179 SEC 10-K filings |
| Companies | 87 (S&P 500 representatives) |
| Sectors | 12 (GICS classification) |
| Fiscal years | 2022, 2023, 2024 |
| Train / Val / Test | 1,600 / 252 / 475 |
| Split strategy | Company-disjoint |
| License | CC BY 4.0 |

---

## Motivation

Existing financial QA benchmarks each cover only a subset of the challenges faced by real-world financial RAG systems:

| Benchmark | Scale | Cross-Doc | Cross-Year | Hybrid-Modal | Difficulty Tiers |
|-----------|------:|:---------:|:----------:|:------------:|:----------------:|
| FinQA | 8,281 | ✗ | ✗ | Partial | ✗ |
| TAT-QA | 16,552 | ✗ | ✗ | ✓ | ✗ |
| FinanceBench | 150 | Partial | ✗ | Partial | ✗ |
| DocFinQA | 7,437 | ✗ | ✗ | ✓ | ✗ |
| ConvFinQA | 3,892 | ✗ | ✗ | Partial | ✗ |
| **Multi-Doc-2025** | **2,327** | **✓** | **✓** | **✓** | **✓ (L1/L2/L3)** |

Multi-Doc-2025 fills this gap with five orthogonal subsets that stress-test different reasoning capabilities independently and in combination.

---

## Dataset Statistics

### Subset Distribution

| Subset | Name | Difficulty | Count | Description |
|--------|------|:----------:|------:|-------------|
| S1 | Single-Doc Fact/Calc | L1 | 800 | Single company, single year, text-based |
| S2 | Single-Doc Table | L1 | 494 | Single company, single year, table-based |
| S3 | Cross-Year Trend | L2 | 243 | Same company, FY2022→2023→2024 |
| S4 | Cross-Company | L2 | 668 | Two companies, same sector, FY2024 |
| S5 | Full-Cross | L3 | 122 | Cross-company + cross-year + hybrid-modal |
| **Total** | | | **2,327** | |

### Intent Distribution

| Intent | Count | % | Description |
|--------|------:|--:|-------------|
| `comparison` | 764 | 32.8% | Side-by-side comparison of two entities |
| `fact` | 672 | 28.9% | Factual retrieval from text |
| `calculation` | 622 | 26.7% | Numerical computation from tables/text |
| `trend` | 269 | 11.6% | Temporal change across fiscal years |

### Structural Properties

| Property | Count | % |
|----------|------:|--:|
| Cross-document (`is_cross_doc`) | 790 | 33.9% |
| Cross-year (`is_cross_year`) | 365 | 15.7% |
| Hybrid-modal (`is_hybrid_modal`) | 1,161 | 49.9% |

### Difficulty Distribution

| Level | Count | % | Description |
|-------|------:|--:|-------------|
| L1 | 1,293 | 55.6% | Single-document, single-modal |
| L2 | 912 | 39.2% | Cross-document or cross-year |
| L3 | 122 | 5.2% | Cross-document + cross-year + hybrid-modal |

---

## Five Subsets

### S1 — Single-Doc Fact/Calculation (L1)

Questions answerable from a single section of a single company's single-year 10-K. Covers both pure text retrieval (`fact`) and numerical computation (`calculation`).

**Example:**
```
Q: What was Apple's total net sales for fiscal year 2022?
A: $394,328 million
```

---

### S2 — Single-Doc Table Reasoning (L1)

Questions requiring multi-row/multi-column table lookups or within-table ratio calculations. All evidence comes from financial tables in Item 7 or Item 8.

**Example:**
```
Q: What was Apple's gross margin percentage for the fiscal year ended September 24, 2022?
A: 43.32%
```

---

### S3 — Cross-Year Trend (L2)

Questions requiring comparison of the same company's metrics across FY2022, FY2023, and FY2024. Tests temporal reasoning and trend identification.

**Example:**
```
Q: How did ExxonMobil's capital expenditures trend from 2022 to 2024?
A: Capital expenditures increased from $16.3B in 2022 to $23.2B in 2024, reflecting accelerated upstream investment.
```

---

### S4 — Cross-Company Comparison (L2)

Questions requiring information from two companies in the same GICS sector, both from FY2024. Tests competitive analysis and relative performance reasoning.

**Example:**
```
Q: Compare JPMorgan and Bank of America's return on equity for FY2024.
A: JPM ROE was 17%, BAC ROE was 9.4%, a difference of 7.6 percentage points.
```

---

### S5 — Full-Cross (L3)

The hardest subset. Questions require evidence from two different companies, across at least two fiscal years, from both text and financial tables simultaneously.

**Example:**
```
Q: How did the revenue gap between Apple and Microsoft evolve from 2022 to 2024?
A: Apple led by $148B in 2022 but the gap narrowed to $146B in 2024 as Microsoft's cloud revenue grew at a 29% CAGR vs Apple's 5% CAGR.
```

---

## Data Format

Each QA pair is a JSON object with the following fields:

```json
{
  "id":                   "md2025_0001",
  "question":             "What was Apple's gross margin percentage for FY2022?",
  "answer":               "43.32%",
  "intent":               "calculation",
  "difficulty":           "L1",
  "subset":               "S2",
  "company":              "AAPL",
  "companies":            ["AAPL"],
  "year":                 "2022",
  "years_required":       ["2022"],
  "sector":               "Information Technology",
  "is_cross_doc":         false,
  "is_cross_year":        false,
  "is_hybrid_modal":      true,
  "requires_calculation": true,
  "evidence_section":     "Item 8. Financial Statements"
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier (`md2025_XXXX`) |
| `question` | string | The question |
| `answer` | string | Ground-truth answer |
| `intent` | string | `calculation` \| `trend` \| `fact` \| `comparison` |
| `difficulty` | string | `L1` \| `L2` \| `L3` |
| `subset` | string | `S1` \| `S2` \| `S3` \| `S4` \| `S5` |
| `company` | string | Primary company ticker (e.g., `AAPL`) |
| `companies` | list[string] | All company tickers required to answer |
| `year` | string | Primary fiscal year |
| `years_required` | list[string] | All fiscal years required to answer |
| `sector` | string | GICS sector of the primary company |
| `is_cross_doc` | bool | Requires evidence from ≥2 documents |
| `is_cross_year` | bool | Requires evidence from ≥2 fiscal years |
| `is_hybrid_modal` | bool | Requires both text and table evidence |
| `requires_calculation` | bool | Answer involves numerical computation |
| `evidence_section` | string | 10-K section(s) containing the evidence |

---

## Files

```
multidoc2025/
├── README.md          # This file
├── train.json         # 1,600 training samples
├── val.json           # 252 validation samples
├── test.json          # 475 test samples
├── datacard.md        # Dataset card (metadata, ethics, limitations)
├── CITATION.cff       # Citation file
└── examples/
    ├── load_dataset.py    # Load and explore the dataset
    └── evaluate.py        # Compute EM / F1 / Exec-Acc metrics
```

---

## Quick Start

```python
import json

# Load splits
train = json.load(open("train.json"))
val   = json.load(open("val.json"))
test  = json.load(open("test.json"))

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Filter by subset
s5_test = [x for x in test if x["subset"] == "S5"]

# Filter by intent
calc_test = [x for x in test if x["intent"] == "calculation"]

# Filter cross-document questions
cross_doc = [x for x in test if x["is_cross_doc"]]
```

Or use the provided helper script:

```bash
python examples/load_dataset.py
```

---

## Evaluation

We report four primary metrics:

| Metric | Description |
|--------|-------------|
| **EM** | Exact Match after answer normalization (lowercase, strip punctuation) |
| **F1** | Token-level F1 score |
| **Exec-Acc** | Numerical execution accuracy (relative tolerance 1×10⁻³) |
| **Hall-Rate** | Hallucination rate — fraction of answers containing unverifiable claims |

Slice metrics are reported for each intent class (`calculation_f1`, `trend_f1`, `fact_f1`, `comparison_f1`), each subset (`subset_S1_f1` … `subset_S5_f1`), and each difficulty level (`difficulty_L1_f1` … `difficulty_L3_f1`).

Use the provided evaluation script:

```bash
python examples/evaluate.py \
    --predictions predictions.json \
    --split test
```

`predictions.json` should be a list of objects with `id` and `prediction` fields:

```json
[
  {"id": "md2025_0001", "prediction": "43.32%"},
  ...
]
```

---

## Construction Pipeline

The full pipeline is reproducible. See the companion paper for detailed construction instructions.

**Summary:**

1. **Download 10-K filings** from SEC EDGAR (179 HTML files, ~2 GB)
2. **Generate candidate QA pairs** using LLM with subset-specific prompts
3. **Quality filter** via rule-based auto-review (removes meta-questions, empty answers, invalid fields)
4. **Finalize splits** with primary-company-disjoint stratified sampling. No primary company appears in more than one split. For cross-company questions, supporting companies may appear across splits because they serve as comparison evidence rather than the primary query entity.

**Estimated reproduction cost:** ~$25–40 in DeepSeek V4-flash API fees.

---

## Data Sources and License

All source documents are publicly available SEC EDGAR filings:

- **Source:** U.S. Securities and Exchange Commission EDGAR database
- **URL:** https://www.sec.gov/cgi-bin/browse-edgar
- **Filing type:** Form 10-K (Annual Report)
- **Fiscal years:** 2022, 2023, 2024

No proprietary data is used. The QA pairs are derived annotations over public filings.

**Dataset license:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

You are free to share and adapt this dataset for any purpose, provided you give appropriate credit.

---

## Citation

If you use Multi-Doc-2025 in your research, please cite:

```bibtex
@article{chen2026hcrag,
  title     = {{HC-RAG}: Hierarchical Cross-Modal Retrieval-Augmented Generation
               for Financial Document Understanding},
  author    = {Chen, Siyuan and Tan, Huaye and Li, You and Liang, Jiajun},
  year      = {2026},
}
```

---

## Contact

For questions or issues, please open an issue on this dataset page.
