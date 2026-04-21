#!/usr/bin/env bash
# run_finetune_eval_t5_ts.sh
# Fine-tune T5-base on each dataset's train split (3 epochs),
# then evaluate on the test split with Pk / WD / F1 / Precision / Recall / Score.
#
# Usage:
#   bash scripts/run_finetune_eval_t5_ts.sh                    # 全量，3 epochs
#   bash scripts/run_finetune_eval_t5_ts.sh --epochs 5         # 自定义轮数
#   bash scripts/run_finetune_eval_t5_ts.sh --datasets vhf tiage
#   bash scripts/run_finetune_eval_t5_ts.sh --max_samples 300  # 快速冒烟测试
#   bash scripts/run_finetune_eval_t5_ts.sh --save_ckpt        # 保存每个数据集的模型

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=============================================="
echo " T5-base Fine-tune + Evaluation (Topic Shift)"
echo " Metrics : Pk  WD  F1  Precision  Recall  Score"
echo " Project : $PROJECT_DIR"
echo " Time    : $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

python3 scripts/finetune_eval_t5_ts.py \
    --model_name_or_path t5-base \
    --datasets all \
    --epochs 3 \
    --batch_size 16 \
    --lr 5e-5 \
    --max_input_len 256 \
    "$@"

echo ""
echo "Done. Results: data/t5_finetune/results.json  |  data/t5_finetune/results.tsv"
