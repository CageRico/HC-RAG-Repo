# HC-RAG

HC-RAG is an implementation of a hierarchical cross-modal retrieval-augmented generation framework for financial question answering over SEC 10-K filings. The code follows the paper setting: a three-level document-section-evidence graph, FinBERT/TAPAS-based text-table representation, four-way query intent routing, and DeepSeek V4-flash as the shared generator (`DS-V4` in the paper tables).

This repository is intended to provide the core reproducibility code. Large datasets, raw filings, indexes, checkpoints, cache files, and generated outputs are not included. Users should prepare those assets locally following the instructions below.

## Repository Scope

Included:

```text
src/                 Core HC-RAG modules
scripts/             Data preparation, indexing, inference, evaluation scripts
tests/               Component tests
examples/            Minimal usage examples
config.yaml          Default paths, model names, and retrieval settings
requirements.txt     Python dependencies
```

Not included:

```text
data/raw/            SEC 10-K HTML filings
data/benchmarks/     Converted public benchmark files
data/multidoc2025/   Full Multi-Doc-2025 JSON data
indexes/             Built hierarchical indexes
checkpoints/         Trained alignment, intent, and fusion checkpoints
cache/               HuggingFace or dataset cache
outputs/             Generated predictions, metrics, and logs
```

The file and directory names are kept in the code paths so that local deployment matches the implementation. Populate them yourself before running full experiments.

## Method Components

HC-RAG contains four main implementation layers:

1. `src/hierarchical_index.py`: typed document, section, text evidence, and table evidence nodes, including sequential section edges inside each 10-K.
2. `src/encoders.py`: FinBERT text encoder, TAPAS table encoder, and contrastive text-table alignment utilities.
3. `src/retriever.py`: document-level, section-level, and evidence-level hierarchical retrieval with query-aware text-table routing and section-neighbor expansion.
4. `src/generator.py`: OpenAI-compatible generation layer using the configured generator. The default is `deepseek-v4-flash`.

The semantic intent labels are:

```text
calculation, trend, fact, comparison
```

Cross-document, cross-year, and hybrid-modal fields are structural evidence attributes, not intent labels.

## Environment

Create or activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

The local development environment used for checks was:

```bash
conda run -n cv_lab python -m compileall src scripts tests
conda run -n cv_lab python -m unittest discover -s tests
```

## Model Configuration

The default generator is configured in `config.yaml`:

```yaml
models:
  generator: "deepseek-v4-flash"
  openai_base_url: "https://api.deepseek.com"
  openai_api_key: ""
```

Set your API key through the environment instead of committing it:

```bash
set OPENAI_API_KEY=your_key_here
```

On Linux or macOS:

```bash
export OPENAI_API_KEY=your_key_here
```

`DS-V4` in paper result tables refers to DeepSeek V4-flash.

For HuggingFace encoders, `config.yaml` defaults to offline loading:

```yaml
models:
  local_files_only: true
```

Before offline runs, download/cache the configured FinBERT and TAPAS models locally. If you want the scripts to download them automatically, set:

```yaml
models:
  local_files_only: false
```

The hierarchical index configuration also includes local structural expansion at the section level:

```yaml
index:
  use_sequential_expansion: true
  sequential_hops: 1
  sequential_neighbors_per_seed: 2
```

When enabled, `scripts/build_index.py` creates explicit `Section -> Section` sequential edges based on each filing's `start_pos` order, and `src/retriever.py` can expand top-ranked sections to their immediate neighbors during L2 retrieval.

## Data Preparation

This repository does not ship the full datasets, raw filings, built indexes, or generated outputs. Multi-Doc-2025 is released separately on HuggingFace:

```text
https://huggingface.co/datasets/Anonymous-Team-HC-RAG/Multi-Doc-2025
```

Download the dataset from HuggingFace and place the files under the expected local paths:

```text
data/raw/                         SEC 10-K HTML filings
data/benchmarks/finqa/
data/benchmarks/tatqa/
data/benchmarks/docfinqa/
data/benchmarks/financebench/
data/multidoc2025/train.json
data/multidoc2025/val.json
data/multidoc2025/test.json
```

The repository may keep empty placeholder directories or metadata files under `data/` so the expected layout is visible, but users must populate the actual dataset files locally before running indexing or evaluation.

Multi-Doc-2025 uses a primary-company-disjoint split: no primary company appears in more than one split. For cross-company questions, supporting companies may appear across splits because they serve as comparison evidence rather than the primary query entity.

You may use the provided scripts as local deployment helpers after configuring paths and credentials:

```bash
python scripts/download_sec_filings.py
python scripts/prepare_datasets.py --output_dir ./data/benchmarks
python scripts/build_multidoc2025.py
```

Depending on your data source and API access, you may need to adapt paths or credentials before running these scripts. If you use the HuggingFace release directly, make sure its split files and original filings are copied or symlinked to the paths expected by `config.yaml` and the scripts.

## Paper-Aligned Training and Inference

After data is available, reproduce the paper pipeline in this order:

1. Build an initial hierarchical index from the local filings:

```bash
python scripts/build_index.py --config config.yaml
```

2. Train the offline text-table alignment model from section-local pairs extracted from the index:

```bash
python scripts/train_align.py
```

3. Rebuild the hierarchical index so cached retrieval embeddings use the trained alignment projection:

```bash
python scripts/build_index.py --config config.yaml
```

4. Train the four-class intent classifier on Multi-Doc-2025:

```bash
python scripts/train_intent.py
```

5. Train the query-aware fusion gate:

```bash
python scripts/train_fusion.py
```

This paper-aligned workflow produces the following checkpoints under `checkpoints/`:

```text
align_checkpoint_best.pt
intent_best.pt
fusion_best.pt
```

The hierarchical index is written under:

```text
indexes/hierarchical_index.pkl
```

The index is intentionally not committed because it depends on local filings, parser output, and model cache state.

## Run Inference

After the paper-aligned checkpoints have been created:

```bash
python scripts/run_inference.py --config config.yaml
```

For end-to-end answer-level evaluation:

```bash
python scripts/run_evaluation.py --dataset multidoc2025 --split test --config config.yaml
```

For baseline evaluation:

```bash
python scripts/run_baselines.py --all_baselines --dataset multidoc2025 --split test --config config.yaml
```

For evidence localization:

```bash
python scripts/run_evidence_eval.py --all_methods --dataset multidoc2025 --split test --config config.yaml
```

Generated predictions, metrics, logs, and figures are written under `outputs/`, which is excluded from version control by default.
Each E2/E3/baseline run also writes a latest `run_metadata.json` plus a timestamped `*_run_metadata_*.json` file so retrieval budgets, evidence budgets, decoding settings, checkpoints, and evaluation cutoffs can be audited after the run.

## Reproducibility Notes

All RAG-style methods use the same generator, decoding setting, and maximum context length. HC-RAG uses an intent-aware evidence-grounded prompt family, while released baselines use a harmonized evidence-grounded prompt and serialization interface. Retrieval budgets and top-k settings are method-specific and should be recorded with each run.

For paper-aligned HC-RAG defaults:

```yaml
index:
  l1_document_k: 5
  l2_section_k: 10
  l3_semantic_k: 20
generation:
  temperature: 0.0
  top_p: 1.0
```

The released HC-RAG code expects the aligned index and the three trained checkpoints above for paper-consistent runs. If the checkpoints are missing, the scripts will still run, but those runs should be treated as non-paper-aligned sanity checks rather than the full reported configuration.

BM25 table-hit evidence metrics should be reported as `--` when comparable `table_id` values are not available. Use `0.00` only when comparable table IDs exist and no gold table is hit.

## Tests

Run:

```bash
python -m compileall src scripts tests
python -m unittest discover -s tests
```

In the original local environment:

```bash
conda run -n cv_lab python -m compileall src scripts tests
conda run -n cv_lab python -m unittest discover -s tests
```
