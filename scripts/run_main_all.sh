#!/usr/bin/env bash

set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

DATASETS=(vhf dialseg711 doc2dial tiage superseg)

MODEL_NAME="${MODEL_NAME:-dud}"
ENCODER="${ENCODER:-roberta-base}"
EPOCHS="${EPOCHS:-50}"
EMB_BATCH="${EMB_BATCH:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SEED="${SEED:-42}"
TOPIC_JSON_PATH="${TOPIC_JSON_PATH:-./data/topic/topic_keywords.json}"
RANK_LOSS_WEIGHT="${RANK_LOSS_WEIGHT:-0.1}"
RANK_MARGIN="${RANK_MARGIN:-0.1}"
RANK_KW_GAP="${RANK_KW_GAP:-0.05}"
EXP_NAME="${EXP_NAME:-main_exp}"
OUT_CSV="${OUT_CSV:-main_exp.csv}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run the main DTS model on all five datasets and save summary metrics to CSV.

Environment variables you can override:
  MODEL_NAME=dud
  ENCODER=roberta-base
  EPOCHS=50
  EMB_BATCH=64
  BATCH_SIZE=8
  SEED=42
  TOPIC_JSON_PATH=./data/topic/topic_keywords.json
  RANK_LOSS_WEIGHT=0.1
  RANK_MARGIN=0.1
  RANK_KW_GAP=0.05
  EXP_NAME=main_exp
  OUT_CSV=main_exp.csv

Example:
  bash scripts/run_main_all.sh
  EXP_NAME=main_exp_v2 EPOCHS=30 bash scripts/run_main_all.sh
EOF
  exit 0
fi

echo "Running main model on datasets: ${DATASETS[*]}"
echo "model=$MODEL_NAME encoder=$ENCODER epochs=$EPOCHS exp_name=$EXP_NAME"

failures=0

for dataset in "${DATASETS[@]}"; do
  echo
  echo "===== [$dataset] main model start ====="
  if ! python "$ROOT_DIR/inference.py" \
    --model_name "$MODEL_NAME" \
    --dataset "$dataset" \
    --encoder "$ENCODER" \
    --epochs "$EPOCHS" \
    --emb_batch "$EMB_BATCH" \
    --batch_size "$BATCH_SIZE" \
    --seed "$SEED" \
    --topic_json_path "$TOPIC_JSON_PATH" \
    --rank_loss_weight "$RANK_LOSS_WEIGHT" \
    --rank_margin "$RANK_MARGIN" \
    --rank_kw_gap "$RANK_KW_GAP" \
    --exp_name "$EXP_NAME"; then
    echo "===== [$dataset] main model failed ====="
    failures=$((failures + 1))
  else
    echo "===== [$dataset] main model done ====="
  fi
done

export ROOT_DIR MODEL_NAME ENCODER EPOCHS EXP_NAME OUT_CSV

python - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["ROOT_DIR"])
model_name = os.environ["MODEL_NAME"]
encoder = os.environ["ENCODER"]
epochs = int(os.environ["EPOCHS"])
exp_name = os.environ["EXP_NAME"]
out_csv = root / os.environ["OUT_CSV"]

datasets = ["vhf", "dialseg711", "doc2dial", "tiage", "superseg"]

rows = []
for dataset in datasets:
    result_path = root / "checkpoints" / dataset / exp_name / "results.json"
    row = {
        "dataset": dataset,
        "model_name": model_name,
        "encoder": encoder,
        "epochs": epochs,
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
        row.update(
            {
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
            "model_name",
            "encoder",
            "epochs",
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
