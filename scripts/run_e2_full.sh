#!/usr/bin/env bash
# Full serial E2 evaluation script.
# Usage: bash scripts/run_e2_full.sh
# Re-running after interruption skips completed combinations (--resume).
# workers=2 keeps memory usage manageable.

set -e
PYTHON=/e/Anaconda3/envs/cv_lab/python.exe
WORKERS=2
LOG=./outputs/e2_run.log

mkdir -p ./outputs
echo "===== E2 started: $(date) =====" | tee -a $LOG

# -------------------------------------------------------
# Stage 1: 9 baselines x 4 datasets.
# Datasets are ordered roughly from small to large; financebench warms up first.
# -------------------------------------------------------

DATASETS="financebench finqa docfinqa tatqa"
BASELINES="bm25 dpr contriever vanilla_rag self_rag graphrag raptor tat_llm tapex_rag"

for ds in $DATASETS; do
    for bl in $BASELINES; do
        echo "----- Baseline=$bl Dataset=$ds $(date) -----" | tee -a $LOG
        $PYTHON scripts/run_baselines.py \
            --baseline $bl \
            --dataset $ds \
            --split test \
            --workers $WORKERS \
            --resume \
            2>&1 | tee -a $LOG
        echo "Done: $bl/$ds $(date)" | tee -a $LOG
    done
done

echo "===== Baselines completed: $(date) =====" | tee -a $LOG

# -------------------------------------------------------
# Stage 2: HC-RAG x 4 datasets.
# -------------------------------------------------------

for ds in $DATASETS; do
    echo "----- HC-RAG Dataset=$ds $(date) -----" | tee -a $LOG
    $PYTHON scripts/run_evaluation.py \
        --dataset $ds \
        --split test \
        --workers $WORKERS \
        2>&1 | tee -a $LOG
    echo "Done: hcrag/$ds $(date)" | tee -a $LOG
done

echo "===== E2 completed: $(date) =====" | tee -a $LOG
echo "Summary: ./outputs/all_results.csv"
