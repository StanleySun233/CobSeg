#!/usr/bin/env bash
# run_eval_t5_nlg.sh
# Zero-shot T5-base evaluation on all five NLG datasets.
#
# Usage:
#   bash scripts/run_eval_t5_nlg.sh                    # 默认全量评测
#   bash scripts/run_eval_t5_nlg.sh --max_samples 200  # 快速冒烟测试
#   bash scripts/run_eval_t5_nlg.sh --datasets vhf tiage
#
# 所有额外参数都会透传给 eval_t5_nlg.py。

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo " T5-base Zero-Shot NLG Evaluation"
echo " Project : $PROJECT_DIR"
echo " Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

python3 scripts/eval_t5_nlg.py \
    --model_name_or_path t5-base \
    --datasets all \
    --batch_size 64 \
    --max_input_len 512 \
    --max_new_tokens 64 \
    --num_beams 5 \
    "$@"

echo ""
echo "Done. Results written to data/t5_nlg/results.json and data/t5_nlg/results.tsv"
