#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(vhf dialseg711 doc2dial tiage superseg)

ENCODER="${ENCODER:-roberta-base}"
MODE="${MODE:-NSP}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-16}"
LR="${LR:-2e-5}"
SEED="${SEED:-42}"
MAX_LENGTH="${MAX_LENGTH:-128}"
ALPHA_LOWER="${ALPHA_LOWER:--2.0}"
ALPHA_UPPER="${ALPHA_UPPER:-2.0}"
ALPHA_STEP="${ALPHA_STEP:-0.1}"
NUM_SAMPLES="${NUM_SAMPLES:--1}"
EXP_NAME="${EXP_NAME:-nsp_texttiling_train}"
OUT_CSV="${OUT_CSV:-nsp_train.csv}"
BALANCE_FLAG="${BALANCE_FLAG:-0}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run supervised NSP TextTiling on all five datasets once and save summary metrics to CSV.

Environment variables you can override:
  ENCODER=roberta-base
  MODE=NSP
  EPOCHS=5
  BATCH_SIZE=16
  LR=2e-5
  SEED=42
  MAX_LENGTH=128
  ALPHA_LOWER=-2.0
  ALPHA_UPPER=2.0
  ALPHA_STEP=0.1
  NUM_SAMPLES=-1
  EXP_NAME=nsp_texttiling_train
  OUT_CSV=nsp_train.csv
  BALANCE_FLAG=0

Example:
  bash scripts/run_nsp_train_all.sh
  EPOCHS=8 BALANCE_FLAG=1 bash scripts/run_nsp_train_all.sh
EOF
  exit 0
fi

echo "Running supervised NSP on datasets: ${DATASETS[*]}"
echo "encoder=$ENCODER mode=$MODE epochs=$EPOCHS exp_name=$EXP_NAME balance=$BALANCE_FLAG"

failures=0

for dataset in "${DATASETS[@]}"; do
  echo
  echo "===== [$dataset] NSP train start ====="

  cmd=(
    python -m model.nsp_texttiling
    --dataset "$dataset"
    --encoder "$ENCODER"
    --mode "$MODE"
    --epochs "$EPOCHS"
    --batch_size "$BATCH_SIZE"
    --lr "$LR"
    --seed "$SEED"
    --max_length "$MAX_LENGTH"
    --alpha_lower "$ALPHA_LOWER"
    --alpha_upper "$ALPHA_UPPER"
    --alpha_step "$ALPHA_STEP"
    --num_samples "$NUM_SAMPLES"
    --exp_name "$EXP_NAME"
  )

  if [[ "$BALANCE_FLAG" == "1" ]]; then
    cmd+=(--balance)
  fi

  if ! "${cmd[@]}"; then
    echo "===== [$dataset] NSP train failed ====="
    failures=$((failures + 1))
  else
    echo "===== [$dataset] NSP train done ====="
  fi
done

export ROOT_DIR ENCODER MODE EPOCHS EXP_NAME OUT_CSV BALANCE_FLAG

python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
encoder = os.environ["ENCODER"]
mode = os.environ["MODE"]
epochs = int(os.environ["EPOCHS"])
exp_name = os.environ["EXP_NAME"]
out_csv = root / os.environ["OUT_CSV"]
balance_flag = os.environ["BALANCE_FLAG"]

datasets = ["vhf", "dialseg711", "doc2dial", "tiage", "superseg"]
encoder_stem = encoder.replace("/", "_")

rows = []
for dataset in datasets:
    result_path = root / "checkpoints" / "bert-finetune" / dataset / encoder_stem / exp_name / "results.json"
    row = {
        "dataset": dataset,
        "encoder": encoder,
        "mode": mode,
        "epochs": epochs,
        "balance": balance_flag,
        "best_epoch": "",
        "best_alpha": "",
        "best_dev_score": "",
        "PK": "",
        "WD": "",
        "F1": "",
        "Precision": "",
        "Recall": "",
        "Score": "",
        "result_json": str(result_path.relative_to(root)),
        "status": "missing",
    }

    if result_path.exists():
        data = json.loads(result_path.read_text(encoding="utf-8"))
        metrics = data.get("metrics_test", {})
        training = data.get("training", {})
        row.update(
            {
                "best_epoch": training.get("best_epoch", ""),
                "best_alpha": data.get("best_alpha", ""),
                "best_dev_score": training.get("best_dev_score", ""),
                "PK": metrics.get("PK", ""),
                "WD": metrics.get("WD", ""),
                "F1": metrics.get("F1", ""),
                "Precision": metrics.get("Precision", ""),
                "Recall": metrics.get("Recall", ""),
                "Score": metrics.get("Score", ""),
                "status": "ok",
            }
        )

    rows.append(row)

with out_csv.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "dataset",
            "encoder",
            "mode",
            "epochs",
            "balance",
            "best_epoch",
            "best_alpha",
            "best_dev_score",
            "PK",
            "WD",
            "F1",
            "Precision",
            "Recall",
            "Score",
            "result_json",
            "status",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved summary CSV to {out_csv}")
PY

if [[ "$failures" -gt 0 ]]; then
  echo "Finished with $failures failed dataset run(s). Summary CSV still generated: $OUT_CSV"
  exit 1
fi

echo "All dataset runs finished successfully. Summary CSV: $OUT_CSV"
