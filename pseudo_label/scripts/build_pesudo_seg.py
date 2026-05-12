from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)

from utils.metrics import evaluate_wd_pk_f1
from utils.dts_utils import segments_to_boundaries
from utils.texttiling_baseline import flatten_segment_samples, segment_texts
from utils.pesudo_common import (
    save_json,
    save_jsonl,
    sample_train_dialogues,
    split_dataset,
    load_dataset,
)
from utils.tet_nsp import cache_nsp_depths, load_nsp_model, texttiling_segment
from utils.utils import resolve_dataset_path


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_pesudo_seg_outputs(
    *,
    output_dir: Path,
    dataset_path: str,
    seed: int,
    sample_size: int,
    alpha: float,
    full_pesudo_archives: list[dict],
    train_dialogues: list[dict],
    device: str,
    nsp_model: str,
    alpha_scores: list[dict],
    valid_metrics: dict | None,
) -> None:
    pesudo_by_id = {str(item["source_dial_id"]): item for item in full_pesudo_archives}
    sampled_train = sample_train_dialogues(train_dialogues, seed, sample_size)
    sampled_ids = {str(item["dial_id"]) for item in sampled_train}
    pesudo_archives = [item for item in full_pesudo_archives if str(item["source_dial_id"]) in sampled_ids]
    flat_segment_samples = []
    for item in pesudo_archives:
        flat_segment_samples.extend(
            flatten_segment_samples(
                str(item["source_dial_id"]),
                segment_texts(item["utterances"], item["pesudo_seg"]),
            )
        )
    sampled_train_cut = []
    for dialogue in sampled_train:
        cut_item = dict(dialogue)
        cut_item["segments"] = pesudo_by_id[str(dialogue["dial_id"])]["pesudo_seg"]
        sampled_train_cut.append(cut_item)
    all_segment_lengths = [len(seg["utterances"]) for seg in flat_segment_samples if seg["utterances"]]
    avg_turn = max(int(np.floor(np.mean(all_segment_lengths))) if all_segment_lengths else 1, 1)

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "pesudo_seg_alpha_search.json",
        {
            "dataset": dataset_path,
            "seed": seed,
            "sample_size": sample_size,
            "alpha": alpha,
            "search_grid": alpha_scores,
            "device": device,
            "nsp_model": nsp_model,
        },
    )
    save_json(output_dir / "pesudo_seg_full_train.json", full_pesudo_archives)
    save_json(output_dir / "pesudo_seg_sampled_train.json", sampled_train_cut)
    save_json(output_dir / f"pesudo_seg_sampled_train_{sample_size}.json", sampled_train_cut)
    save_jsonl(output_dir / "pesudo_seg_samples.jsonl", flat_segment_samples)
    if valid_metrics is not None:
        save_json(output_dir / "pesudo_seg_valid_metrics.json", {"alpha": alpha, **valid_metrics})
    save_json(
        output_dir / "pesudo_seg_manifest.json",
        {
            "dataset": dataset_path,
            "full_train_dialogue_count": len(train_dialogues),
            "sampled_train_ids": [item["dial_id"] for item in sampled_train],
            "selected_train_ids": [item["dial_id"] for item in sampled_train],
            "sample_size": sample_size,
            "alpha": alpha,
            "avg_turn": avg_turn,
            "full_pesudo_dialogue_count": len(full_pesudo_archives),
            "selected_pesudo_dialogue_count": len(pesudo_archives),
            "selected_segment_count": len(flat_segment_samples),
            "device": device,
            "nsp_model": nsp_model,
        },
    )
    print(
        f"Wrote pesudo seg artifacts to {output_dir} "
        f"(selected {len(sampled_train)} from {len(train_dialogues)} train dialogues)"
    )


def build_pesudo_seg_outputs_incremental(
    *,
    output_dir: Path,
    dataset_path: str,
    seed: int,
    sample_size: int,
    train_dialogues: list[dict],
    inherit_artifact_dir: Path,
) -> bool:
    inherit_manifest_path = inherit_artifact_dir / "pesudo_seg_manifest.json"
    inherit_full_pesudo_path = inherit_artifact_dir / "pesudo_seg_full_train.json"
    if not inherit_manifest_path.exists() or not inherit_full_pesudo_path.exists():
        return False
    inherit_manifest = load_json(inherit_manifest_path)
    inherited_ids = [str(item) for item in inherit_manifest.get("sampled_train_ids", [])]
    inherited_size = len(inherited_ids)
    if inherited_size >= sample_size:
        return False
    if inherit_manifest.get("dataset") != dataset_path:
        return False
    train_by_id = {str(item["dial_id"]): item for item in train_dialogues}
    train_order = {str(item["dial_id"]): idx for idx, item in enumerate(train_dialogues)}
    if any(dial_id not in train_by_id for dial_id in inherited_ids):
        return False

    inherited_id_set = set(inherited_ids)
    remaining_dialogues = [item for item in train_dialogues if str(item["dial_id"]) not in inherited_id_set]
    extra_size = sample_size - inherited_size
    if extra_size > len(remaining_dialogues):
        raise SystemExit(
            f"Only {len(train_dialogues)} training dialogues available; cannot sample {sample_size}."
        )
    extra_dialogues = sample_train_dialogues(remaining_dialogues, seed, extra_size)
    combined_dialogues = [train_by_id[dial_id] for dial_id in inherited_ids] + extra_dialogues
    sampled_train = sorted(combined_dialogues, key=lambda row: train_order[str(row["dial_id"])])
    sampled_ids = {str(item["dial_id"]) for item in sampled_train}

    full_pesudo_archives = load_json(inherit_full_pesudo_path)
    pesudo_by_id = {str(item["source_dial_id"]): item for item in full_pesudo_archives}
    if any(dial_id not in pesudo_by_id for dial_id in sampled_ids):
        missing_ids = sorted(dial_id for dial_id in sampled_ids if dial_id not in pesudo_by_id)
        raise SystemExit(f"Missing pesudo seg records for dialogue ids: {missing_ids[:10]}")

    sampled_train_cut = []
    flat_segment_samples = []
    pesudo_archives = [pesudo_by_id[str(item["dial_id"])] for item in sampled_train]
    for dialogue in sampled_train:
        cut_item = dict(dialogue)
        cut_item["segments"] = pesudo_by_id[str(dialogue["dial_id"])]["pesudo_seg"]
        sampled_train_cut.append(cut_item)
    for item in pesudo_archives:
        flat_segment_samples.extend(
            flatten_segment_samples(
                str(item["source_dial_id"]),
                segment_texts(item["utterances"], item["pesudo_seg"]),
            )
        )
    all_segment_lengths = [len(seg["utterances"]) for seg in flat_segment_samples if seg["utterances"]]
    avg_turn = max(int(np.floor(np.mean(all_segment_lengths))) if all_segment_lengths else 1, 1)

    alpha_search = load_json(inherit_artifact_dir / "pesudo_seg_alpha_search.json")
    alpha = float(alpha_search["alpha"])
    valid_metrics = None
    valid_metrics_path = inherit_artifact_dir / "pesudo_seg_valid_metrics.json"
    if valid_metrics_path.exists():
        valid_metrics_raw = load_json(valid_metrics_path)
        valid_metrics = {
            "WD": float(valid_metrics_raw.get("WD", 0.0)),
            "PK": float(valid_metrics_raw.get("PK", 0.0)),
            "F1": float(valid_metrics_raw.get("F1", 0.0)),
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        output_dir / "pesudo_seg_alpha_search.json",
        {
            **alpha_search,
            "dataset": dataset_path,
            "seed": seed,
            "sample_size": sample_size,
        },
    )
    save_json(output_dir / "pesudo_seg_full_train.json", full_pesudo_archives)
    save_json(output_dir / "pesudo_seg_sampled_train.json", sampled_train_cut)
    save_json(output_dir / f"pesudo_seg_sampled_train_{sample_size}.json", sampled_train_cut)
    save_jsonl(output_dir / "pesudo_seg_samples.jsonl", flat_segment_samples)
    if valid_metrics is not None:
        save_json(output_dir / "pesudo_seg_valid_metrics.json", {"alpha": alpha, **valid_metrics})
    save_json(
        output_dir / "pesudo_seg_manifest.json",
        {
            "dataset": dataset_path,
            "full_train_dialogue_count": len(train_dialogues),
            "sampled_train_ids": [item["dial_id"] for item in sampled_train],
            "selected_train_ids": [item["dial_id"] for item in sampled_train],
            "sample_size": sample_size,
            "alpha": alpha,
            "avg_turn": avg_turn,
            "full_pesudo_dialogue_count": len(full_pesudo_archives),
            "selected_pesudo_dialogue_count": len(pesudo_archives),
            "selected_segment_count": len(flat_segment_samples),
            "device": str(alpha_search.get("device", "unknown")),
            "nsp_model": str(alpha_search.get("nsp_model", "")),
            "inherited_pesudo_dir": str(inherit_artifact_dir),
            "inherited_sample_size": inherited_size,
            "incremental_added_count": extra_size,
        },
    )
    print(
        f"Wrote pesudo seg artifacts to {output_dir} "
        f"(inherited {inherited_size} and added {extra_size} dialogues)"
    )
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset/dialseg711.json", help="Dataset path or alias")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--alpha-min", type=float, default=-2.0)
    parser.add_argument("--alpha-max", type=float, default=2.0)
    parser.add_argument("--alpha-step", type=float, default=0.1)
    parser.add_argument("--output-dir", default="data/dataset/dialseg711_pesudo_100_artifacts")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--nsp-model", default="bert-base-uncased")
    parser.add_argument("--inherit-pesudo-dir", default="")
    args = parser.parse_args()

    dataset_path = resolve_dataset_path(args.dataset)
    data = load_dataset(dataset_path)
    split_map = split_dataset(data)
    train_dialogues = split_map["train"]
    valid_dialogues = split_map["valid"]
    output_dir = Path(args.output_dir)

    if len(train_dialogues) < args.sample_size:
        raise SystemExit(
            f"Only {len(train_dialogues)} training dialogues available; cannot sample {args.sample_size}."
        )

    if args.inherit_pesudo_dir:
        inherited = build_pesudo_seg_outputs_incremental(
            output_dir=output_dir,
            dataset_path=dataset_path,
            seed=args.seed,
            sample_size=args.sample_size,
            train_dialogues=train_dialogues,
            inherit_artifact_dir=Path(args.inherit_pesudo_dir),
        )
        if inherited:
            return

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    tokenizer, model, use_nsp_head = load_nsp_model(args.nsp_model, device)
    valid_cache = cache_nsp_depths(
        valid_dialogues,
        tokenizer,
        model,
        device,
        batch_size=64,
        use_nsp_head=use_nsp_head,
    )
    train_cache = cache_nsp_depths(
        train_dialogues,
        tokenizer,
        model,
        device,
        batch_size=64,
        use_nsp_head=use_nsp_head,
    )

    if args.alpha is None:
        alpha_grid = np.round(np.arange(args.alpha_min, args.alpha_max, args.alpha_step), 10).tolist()
        alpha_scores: list[dict] = []
        best_alpha = None
        best_pk = None
        for alpha in tqdm(alpha_grid, desc="alpha search", unit="alpha"):
            wd_vals = []
            pk_vals = []
            f1_vals = []
            for item in tqdm(valid_cache, desc=f"valid@{alpha:.1f}", leave=False, unit="dlg"):
                dialogue = item["dialogue"]
                pred_boundaries = texttiling_segment(item["depth"], item["n"], alpha)
                gold_boundaries = segments_to_boundaries(dialogue["segments"])
                wd, pk, f1 = evaluate_wd_pk_f1(pred_boundaries, gold_boundaries)
                wd_vals.append(float(wd))
                pk_vals.append(float(pk))
                f1_vals.append(float(f1))
            if not pk_vals:
                continue
            mean_wd = float(np.mean(wd_vals))
            mean_pk = float(np.mean(pk_vals))
            mean_f1 = float(np.mean(f1_vals))
            alpha_scores.append({"alpha": alpha, "wd": mean_wd, "pk": mean_pk, "f1": mean_f1})
            if best_pk is None or mean_pk < best_pk:
                best_pk = mean_pk
                best_alpha = alpha
        if best_alpha is None:
            raise SystemExit("Failed to search alpha on valid split.")
        alpha = float(best_alpha)
    else:
        alpha = float(args.alpha)
        alpha_scores = []

    output_dir.mkdir(parents=True, exist_ok=True)
    if alpha_scores:
        best_row = min(alpha_scores, key=lambda row: row["pk"])
        print(
            "[alpha search] "
            f"best_alpha={best_row['alpha']:.2f} "
            f"valid_PK={best_row['pk']:.4f} "
            f"valid_WD={best_row['wd']:.4f} "
            f"valid_F1={best_row['f1']:.4f}"
        )

    full_pesudo_archives: list[dict] = []
    full_flat_segment_samples: list[dict] = []
    for item in tqdm(train_cache, desc="pseudo cut", unit="dlg"):
        dialogue = item["dialogue"]
        depth = item["depth"]
        n = item["n"]
        boundaries = texttiling_segment(depth, n, alpha)
        result_segments: list[int] = []
        count = 0
        for b in boundaries:
            count += 1
            if b == 1:
                result_segments.append(count)
                count = 0
        if count > 0:
            result_segments.append(count)
        result_scores = [float(v) for v in depth.tolist()]
        result_threshold = float(depth.mean() + float(alpha) * depth.std()) if depth.size else 0.0
        full_pesudo_archives.append(
            {
                "source_dial_id": dialogue["dial_id"],
                "set": dialogue.get("set", "train"),
                "utterances": dialogue["utterances"],
                "gold_segments": dialogue["segments"],
                "pesudo_seg": result_segments,
                "pesudo_label": boundaries,
                "depth_scores": result_scores,
                "threshold": result_threshold,
            }
        )
        full_flat_segment_samples.extend(
            flatten_segment_samples(str(dialogue["dial_id"]), segment_texts(dialogue["utterances"], result_segments))
        )

    valid_metrics = None
    if args.alpha is None:
        valid_metrics = {"WD": 0.0, "PK": 0.0, "F1": 0.0}
        wd_vals = []
        pk_vals = []
        f1_vals = []
        for item in valid_cache:
            dialogue = item["dialogue"]
            pred_boundaries = texttiling_segment(item["depth"], item["n"], alpha)
            gold_boundaries = segments_to_boundaries(dialogue["segments"])
            wd, pk, f1 = evaluate_wd_pk_f1(pred_boundaries, gold_boundaries)
            wd_vals.append(float(wd))
            pk_vals.append(float(pk))
            f1_vals.append(float(f1))
        if wd_vals:
            valid_metrics = {
                "WD": float(np.mean(wd_vals)),
                "PK": float(np.mean(pk_vals)),
                "F1": float(np.mean(f1_vals)),
            }
        print(
            "[valid metrics] "
            f"alpha={alpha:.2f} "
            f"PK={valid_metrics['PK']:.4f} "
            f"WD={valid_metrics['WD']:.4f} "
            f"F1={valid_metrics['F1']:.4f}"
        )
    build_pesudo_seg_outputs(
        output_dir=output_dir,
        dataset_path=dataset_path,
        seed=args.seed,
        sample_size=args.sample_size,
        alpha=alpha,
        full_pesudo_archives=full_pesudo_archives,
        train_dialogues=train_dialogues,
        device=str(device),
        nsp_model=args.nsp_model,
        alpha_scores=alpha_scores,
        valid_metrics=valid_metrics,
    )


if __name__ == "__main__":
    main()
