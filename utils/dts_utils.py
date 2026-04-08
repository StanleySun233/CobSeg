import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from utils.metrics import evaluate_segmentation


def run_checkpoint_dir(model_file: str, resolved_dataset_path: str, exp_name: str) -> Path:
    sub = Path(resolved_dataset_path).stem
    root = Path(model_file).resolve().parent / "checkpoints"
    return root / sub / exp_name


def dialogues_used_for_stream(dialogues: list, max_utterances: int) -> list:
    return [d for d in dialogues if len(d.utterances[:max_utterances]) > 0]


def segments_to_boundaries(segments: list[int]) -> list[int]:
    boundaries: list[int] = []
    for seg_idx, seg_len in enumerate(segments):
        is_last = seg_idx == len(segments) - 1
        for pos in range(seg_len):
            if pos == seg_len - 1 and not is_last:
                boundaries.append(1)
            else:
                boundaries.append(0)
    return boundaries


def evaluate_all(all_preds: list[list[int]], all_labels: list[list[int]]) -> dict:
    pk_list, wd_list, f1_list = [], [], []
    all_pred_flat, all_true_flat = [], []

    for preds, labels in zip(all_preds, all_labels):
        m = evaluate_segmentation(labels, preds)
        pk_list.append(m["PK"])
        wd_list.append(m["WD"])
        f1_list.append(m["F1"])
        all_pred_flat.extend(preds)
        all_true_flat.extend(labels)

    precision = precision_score(all_true_flat, all_pred_flat, pos_label=1, zero_division=0)
    recall = recall_score(all_true_flat, all_pred_flat, pos_label=1, zero_division=0)

    return {
        "PK": float(np.mean(pk_list)),
        "WD": float(np.mean(wd_list)),
        "F1": float(np.mean(f1_list)),
        "Precision": float(precision),
        "Recall": float(recall),
    }


def print_metrics(metrics: dict, prefix: str = ""):
    tag = f"[{prefix}] " if prefix else ""
    print(
        f"{tag}PK={metrics['PK']:.4f}  WD={metrics['WD']:.4f}  "
        f"F1={metrics['F1']:.4f}  P={metrics['Precision']:.4f}  R={metrics['Recall']:.4f}"
    )


def save_sample_predictions(
    dialogues: list,
    all_preds: list[list[int]],
    all_labels: list[list[int]],
    out_path: str | Path,
    n: int = 5,
    seed: int = 42,
) -> None:
    if n == -1:
        indices = list(range(len(dialogues)))
    else:
        rng = random.Random(seed)
        indices = rng.sample(range(len(dialogues)), min(n, len(dialogues)))

    rows = []
    for idx in indices:
        dial = dialogues[idx]
        n_lab = len(all_labels[idx])
        utts = dial.utterances[:n_lab]
        for utt_idx, (utt, true, pred) in enumerate(zip(utts, all_labels[idx], all_preds[idx])):
            rows.append(
                {
                    "dial_id": dial.dial_id,
                    "utt_idx": utt_idx,
                    "utterance": utt,
                    "true_label": true,
                    "pred_label": pred,
                    "correct": int(true == pred),
                }
            )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    if n == -1:
        print(f"Full test predictions ({len(indices)} dialogues) saved to {out_path}")
    else:
        print(f"Sample predictions ({len(indices)} dialogues) saved to {out_path}")
