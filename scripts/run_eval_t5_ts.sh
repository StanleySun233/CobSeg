#!/usr/bin/env bash
# run_eval_t5_ts.sh
# Zero-shot T5-base topic-shift detection on all five datasets.
# Metrics: Pk  WD  F1  Precision  Recall  Score
#
# Usage:
#   bash scripts/run_eval_t5_ts.sh                     # 默认全量评测
#   bash scripts/run_eval_t5_ts.sh --max_samples 200   # 快速冒烟测试
#   bash scripts/run_eval_t5_ts.sh --datasets vhf tiage

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo " T5-base Zero-Shot Topic-Shift Evaluation"
echo " Metrics : Pk  WD  F1  Precision  Recall  Score"
echo " Project : $PROJECT_DIR"
echo " Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

python3 scripts/eval_t5_ts.py \
    --model_name_or_path t5-base \
    --datasets all \
    --batch_size 64 \
    --max_input_len 512 \
    --max_new_tokens 4 \
    --num_beams 2 \
    "$@"

echo ""
echo "Done. Results: data/t5_eval/results.json  |  data/t5_eval/results.tsv"
