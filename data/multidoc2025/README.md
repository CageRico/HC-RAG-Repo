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
  - cross-company
  - cross-year
  - cross-modal
  - table-qa
  - financial-qa
  - rag-benchmark
pretty_name: Multi-Doc-2025
size_categories:
  - 1K<n<10K
---

# Dataset Card for Multi-Doc-2025

## Dataset Summary

**Multi-Doc-2025** is a financial question-answering benchmark built from SEC Form 10-K annual reports of S&P 500 companies. It is designed to evaluate retrieval-augmented generation (RAG) and financial QA systems under three reasoning settings that are not jointly covered by existing financial QA benchmarks: **cross-company reasoning**, **cross-year reasoning**, and **hybrid-modal reasoning** over both text and tables.

The dataset contains **2,327 QA pairs** across five orthogonal subsets (S1--S5) and three difficulty levels (L1--L3). The questions are derived from **179 SEC 10-K filings** of **87 S&P 500 representative companies** across **12 GICS sectors** and **three fiscal years (2022--2024)**. The official train/validation/test split is **primary-company-disjoint**.

No primary company appears in more than one split. For cross-company questions, supporting companies may appear across splits because they serve as comparison evidence rather than the primary query entity.

Multi-Doc-2025 is intended to support research on evidence-grounded financial question answering, multi-document retrieval, temporal reasoning over annual reports, and text-table reasoning in financial documents.

## Motivation

Existing financial QA benchmarks each cover only a subset of the challenges faced by real-world financial RAG systems. Some focus on numerical reasoning, some focus on text-table reasoning, and some evaluate hallucination or fact verification. However, real-world financial analysis often requires a system to retrieve and reason over multiple filings, multiple fiscal years, and both narrative text and structured financial tables.

| Benchmark | Scale | Cross-Doc | Cross-Year | Hybrid-Modal | Difficulty Tiers |
|-----------|------:|:---------:|:----------:|:------------:|:----------------:|
| FinQA | 8,281 | ✗ | ✗ | Partial | ✗ |
| TAT-QA | 16,552 | ✗ | ✗ | ✓ | ✗ |
| FinanceBench | 150 | Partial | ✗ | Partial | ✗ |
| DocFinQA | 7,437 | ✗ | ✗ | ✓ | ✗ |
| ConvFinQA | 3,892 | ✗ | ✗ | Partial | ✗ |
| **Multi-Doc-2025** | **2,327** | **✓** | **✓** | **✓** | **✓ (L1/L2/L3)** |

Multi-Doc-2025 fills this gap with five orthogonal subsets that stress-test different financial reasoning capabilities independently and in combination.

## Dataset Details

### Dataset Description

- **Curated by:** Anonymous Authors
- **Language:** English
- **License:** CC BY 4.0
- **Source documents:** Public SEC EDGAR Form 10-K filings
- **Document type:** Annual reports
- **Fiscal years:** 2022, 2023, 2024
- **Domain:** Corporate finance, financial accounting, and annual-report analysis

### Dataset Sources

- **Source filings:** SEC EDGAR Form 10-K annual reports
- **Repository:** Coming soon
- **Paper:** Coming soon

## Intended Uses

### Direct Use

Multi-Doc-2025 is intended for:

- Benchmarking financial QA and RAG systems
- Evaluating multi-document financial reasoning
- Evaluating cross-company and peer-comparison reasoning
- Evaluating cross-year and temporal financial reasoning
- Evaluating hybrid-modal reasoning over text and tables
- Studying evidence-grounded generation and hallucination in financial QA
- Developing and testing financial-domain NLP and information retrieval systems

### Out-of-Scope Use

Multi-Doc-2025 should **not** be used for:

- Making real-world investment decisions
- Providing financial advice
- Inferring current company performance from historical filings
- Training general-purpose language models without considering domain bias and data provenance
- Evaluating non-English financial document understanding

The dataset is built from historical filings and may not reflect current corporate performance, restatements, regulatory changes, or later market conditions.

## Dataset Structure

### Data Splits

| Split | Samples |
|-------|--------:|
| train | 1,600 |
| val | 252 |
| test | 475 |
| **Total** | **2,327** |

The splits are **primary-company-disjoint**: all QA pairs for the same primary company are assigned to exactly one split. No primary company appears in more than one split. For cross-company questions, supporting companies may appear across splits because they serve as comparison evidence rather than the primary query entity. This design reduces primary-company leakage and tests whether models generalize to unseen primary query entities.

### Dataset Statistics

| Property | Value |
|----------|------:|
| Total QA pairs | 2,327 |
| Source documents | 179 SEC 10-K filings |
| Companies | 87 S&P 500 representatives |
| Sectors | 12 GICS sectors |
| Fiscal years | 2022, 2023, 2024 |
| Train / Val / Test | 1,600 / 252 / 475 |
| Split strategy | Primary-company-disjoint |
| License | CC BY 4.0 |

### Subset Distribution

| Subset | Name | Difficulty | Count | Description |
|--------|------|:----------:|------:|-------------|
| S1 | Single-Doc Fact/Calc | L1 | 800 | Single company, single year, primarily text-based fact or calculation questions |
| S2 | Single-Doc Table | L1 | 494 | Single company, single year, table-based questions requiring table lookup or within-table computation |
| S3 | Cross-Year Trend | L2 | 243 | Same company across FY2022--FY2024; focuses on temporal trend reasoning |
| S4 | Cross-Company | L2 | 668 | Two companies from the same sector, usually in FY2024; focuses on peer comparison |
| S5 | Full-Cross | L3 | 122 | Cross-company + cross-year + hybrid-modal reasoning |
| **Total** | | | **2,327** | |

### Intent Distribution

The `intent` field captures the **semantic reasoning intent** of a question. It is distinct from structural attributes such as cross-document or hybrid-modal evidence.

| Intent | Count | % | Description |
|--------|------:|--:|-------------|
| `comparison` | 764 | 32.8% | Side-by-side comparison of two entities, metrics, companies, or periods |
| `fact` | 672 | 28.9% | Factual retrieval from text or tables |
| `calculation` | 622 | 26.7% | Numerical computation from financial values in text or tables |
| `trend` | 269 | 11.6% | Temporal change across fiscal years |

### Structural Properties

The following fields describe where the supporting evidence is located and what kind of evidence structure the question requires.

| Property | Count | % |
|----------|------:|--:|
| Cross-document (`is_cross_doc`) | 790 | 33.9% |
| Cross-year (`is_cross_year`) | 365 | 15.7% |
| Hybrid-modal (`is_hybrid_modal`) | 1,161 | 49.9% |

### Difficulty Distribution

| Level | Count | % | Description |
|-------|------:|--:|-------------|
| L1 | 1,293 | 55.6% | Single-document and relatively localized evidence |
| L2 | 912 | 39.2% | Cross-document or cross-year reasoning |
| L3 | 122 | 5.2% | Cross-document + cross-year + hybrid-modal reasoning |

## Five Subsets

### S1 — Single-Doc Fact/Calculation (L1)

Questions answerable from a single section of one company's 10-K filing for a single fiscal year. This subset covers both pure fact retrieval and simple numerical computation.

**Example:**

```text
Q: What was Apple's total net sales for fiscal year 2022?
A: $394,328 million
```

### S2 — Single-Doc Table Reasoning (L1)

Questions requiring table lookup, multi-row/multi-column retrieval, or within-table ratio calculation. Evidence usually comes from Item 7 or Item 8 financial tables.

**Example:**

```text
Q: What was Apple's gross margin percentage for the fiscal year ended September 24, 2022?
A: 43.32%
```

### S3 — Cross-Year Trend (L2)

Questions requiring comparison of the same company's metrics across fiscal years 2022, 2023, and 2024. This subset tests temporal reasoning and trend identification.

**Example:**

```text
Q: How did ExxonMobil's capital expenditures trend from 2022 to 2024?
A: Capital expenditures increased from $16.3B in 2022 to $23.2B in 2024, reflecting accelerated upstream investment.
```

### S4 — Cross-Company Comparison (L2)

Questions requiring evidence from two companies in the same GICS sector. This subset tests peer comparison and relative financial performance reasoning.

**Example:**

```text
Q: Compare JPMorgan and Bank of America's return on equity for FY2024.
A: JPM ROE was 17%, BAC ROE was 9.4%, a difference of 7.6 percentage points.
```

### S5 — Full-Cross (L3)

The hardest subset. Questions require evidence across companies, across years, and across modalities. This subset stress-tests multi-document, temporal, and hybrid-modal reasoning simultaneously.

**Example:**

```text
Q: How did the revenue gap between Apple and Microsoft evolve from 2022 to 2024?
A: Apple led by $148B in 2022 but the gap narrowed to $146B in 2024 as Microsoft's cloud revenue grew at a 29% CAGR vs Apple's 5% CAGR.
```

## Data Fields

Each QA pair is a JSON object with the following fields.

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique identifier, e.g., `md2025_0001` |
| `question` | string | Natural-language financial question |
| `answer` | string | Ground-truth answer |
| `intent` | string | One of `calculation`, `trend`, `fact`, `comparison` |
| `difficulty` | string | One of `L1`, `L2`, `L3` |
| `subset` | string | One of `S1`, `S2`, `S3`, `S4`, `S5` |
| `company` | string | Primary company ticker |
| `companies` | list[string] | All company tickers required to answer the question |
| `year` | string | Primary fiscal year |
| `years_required` | list[string] | All fiscal years required to answer the question |
| `sector` | string | GICS sector of the primary company |
| `is_cross_doc` | bool | Whether evidence from two or more documents is required |
| `is_cross_year` | bool | Whether evidence from two or more fiscal years is required |
| `is_hybrid_modal` | bool | Whether both text and table evidence are required |
| `requires_calculation` | bool | Whether the answer involves numerical computation |
| `evidence_section` | string | 10-K section(s) containing the evidence |

### Example Instance

```json
{
  "id": "md2025_0001",
  "question": "What was Apple's gross margin percentage for the fiscal year ended September 24, 2022?",
  "answer": "43.32%",
  "intent": "calculation",
  "difficulty": "L1",
  "subset": "S2",
  "company": "AAPL",
  "companies": ["AAPL"],
  "year": "2022",
  "years_required": ["2022"],
  "sector": "Information Technology",
  "is_cross_doc": false,
  "is_cross_year": false,
  "is_hybrid_modal": true,
  "requires_calculation": true,
  "evidence_section": "Item 8. Financial Statements"
}
```

## Dataset Creation

### Curation Rationale

Existing financial QA benchmarks each cover only part of the challenges faced by real-world financial RAG systems. Some focus on numerical reasoning, some focus on table-text QA, and some evaluate financial hallucination or fact verification. Multi-Doc-2025 was designed to provide a unified benchmark that simultaneously evaluates:

1. cross-company reasoning,
2. cross-year reasoning,
3. hybrid-modal text-table reasoning,
4. explicit difficulty tiers, and
5. slice-wise evaluation by intent, subset, and difficulty.

### Source Data

All source documents are Form 10-K annual reports downloaded from SEC EDGAR. The dataset covers:

- 87 S&P 500 representative companies
- 12 GICS sectors
- Fiscal years 2022, 2023, and 2024
- 179 HTML filings
- Approximately 2 GB of source filings

### QA Generation

Candidate QA pairs were generated using a large language model with subset-specific prompts. The prompts were designed to elicit questions matching the target subset and difficulty level. Each prompt included an excerpt from the relevant 10-K section(s), and generated candidates were linked to metadata such as company, year, sector, subset, and intent.

### Quality Filtering

Generated candidates were filtered using rule-based quality checks, including:

- minimum question length,
- minimum answer length,
- removal of meta-questions that refer to "the provided excerpt" or "the document",
- removal of unanswerable questions,
- validation of intent labels,
- validation of difficulty labels,
- validation of subset labels,
- validation of required metadata fields.

### Expert Review

After rule-based filtering, the remaining QA pairs were reviewed by a finance domain expert, who verified factual correctness against the original SEC filings. The review focused on answer correctness, consistency with the corresponding filing, and validity of the assigned labels.

### Split Construction

The dataset is split into train, validation, and test sets using primary-company-disjoint stratified sampling. All QA pairs for a given primary company are assigned to exactly one split, while subset balance is maintained as much as possible within each split. No primary company appears in more than one split. For cross-company questions, supporting companies may appear across splits because they serve as comparison evidence rather than the primary query entity.

## Evaluation

### Primary Metrics

The official evaluation reports four primary answer-level metrics:

| Metric | Description |
|--------|-------------|
| EM | Exact Match after answer normalization |
| F1 | Token-level F1 score |
| Exec-Acc | Numerical execution accuracy with tolerance `1e-3` |
| Hall-Rate | Fraction of answers containing unverifiable claims |

### Slice Metrics

Slice metrics are reported by:

- intent class: `calculation_f1`, `trend_f1`, `fact_f1`, `comparison_f1`
- subset: `subset_S1_f1`, `subset_S2_f1`, `subset_S3_f1`, `subset_S4_f1`, `subset_S5_f1`
- difficulty: `difficulty_L1_f1`, `difficulty_L2_f1`, `difficulty_L3_f1`

### Prediction Format

Predictions should be provided as a list of JSON objects with `id` and `prediction` fields.

```json
[
  {"id": "md2025_0001", "prediction": "43.32%"}
]
```

Example command:

```bash
python examples/evaluate.py \
  --predictions predictions.json \
  --split test
```

## Files

```text
multidoc2025/
├── README.md
├── train.json
├── val.json
├── test.json
├── datacard.md
├── CITATION.cff
└── examples/
    ├── load_dataset.py
    └── evaluate.py
```

## Quick Start

```python
import json

train = json.load(open("train.json"))
val = json.load(open("val.json"))
test = json.load(open("test.json"))

print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

# Filter by subset
s5_test = [x for x in test if x["subset"] == "S5"]

# Filter by intent
calc_test = [x for x in test if x["intent"] == "calculation"]

# Filter cross-document questions
cross_doc = [x for x in test if x["is_cross_doc"]]
```

## Bias, Risks, and Limitations

### Coverage Bias

The dataset covers S&P 500 companies, which are large-cap U.S. public firms. Performance on this benchmark may not generalize to small-cap firms, private firms, non-U.S. companies, or non-English financial reports.

### Temporal Scope

The source filings cover fiscal years 2022--2024. The dataset should not be used to infer current company performance or current investment conditions.

### Language Scope

All filings and QA pairs are in English. The dataset does not evaluate multilingual financial document understanding.

### Answer Format

Answers are free-form strings. Numerical answers may have multiple valid representations, such as `$394.3 billion` versus `$394,328 million`. The evaluation script normalizes common formats, but edge cases may remain.

### LLM-Generated Candidates

Candidate QA pairs were generated using a large language model and then filtered and reviewed. Although quality-control steps were applied, occasional factual or labeling errors may remain.

### Subset Imbalance

The S5 Full-Cross subset contains 122 samples because its construction requires cross-company, cross-year, and hybrid-modal reasoning simultaneously. Results on S5 should therefore be interpreted with this smaller sample size in mind.

### Not Financial Advice

The dataset is intended for research and benchmarking only. It should not be used as a source of financial advice or investment recommendations.

## License

The dataset is released under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license. Source filings are publicly available SEC EDGAR filings.

## Citation

If you use Multi-Doc-2025 in your research, please cite:

```bibtex
@misc{anonymous2026multidoc2025,
  title  = {Multi-Doc-2025: A Multi-Document Financial Question Answering Benchmark},
  author = {Anonymous Authors},
  year   = {2026},
  note   = {Anonymous dataset release}
}
```
