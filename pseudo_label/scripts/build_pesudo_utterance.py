from __future__ import annotations

import argparse
import json
import math
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys

from dotenv import load_dotenv
from tqdm import tqdm
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)

from utils.pesudo_common import (
    generate_pesudo_dialogue_with_openai,
    get_openai_client,
    load_dataset,
    save_json,
    save_jsonl,
    split_dataset,
)
from utils.utils import base_dataset_name, normalize_dataset_name


DATASET_CONFIG_PATH = ROOT / "data" / "dataset_config.yaml"


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sample_segment_length(
    rng: random.Random,
    *,
    mean: float,
    std: float,
    min_len: int,
    max_len: int,
) -> int:
    if std <= 0:
        return max(min(int(round(mean)), max_len), min_len)
    for _ in range(32):
        sampled = int(round(rng.gauss(mean, std)))
        if min_len <= sampled <= max_len:
            return sampled
    clipped = int(round(rng.gauss(mean, std)))
    return max(min(clipped, max_len), min_len)


def _resolve_default_max_utterances(dataset: str, data: list[dict]) -> int:
    if DATASET_CONFIG_PATH.exists():
        with DATASET_CONFIG_PATH.open("r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        dataset_key = normalize_dataset_name(dataset)
        base_key = base_dataset_name(dataset)
        ds_cfg = config.get(dataset_key) or config.get(base_key) or {}
        configured = int(ds_cfg.get("max_utterances", 0) or 0)
        if configured > 0:
            return configured
    observed = max((len(item.get("utterances", [])) for item in data), default=0)
    if observed > 0:
        return observed
    raise SystemExit("Unable to resolve max_utterances from dataset config or data.")


def _slice_segments(segments: list[int], start: int, end: int) -> list[int]:
    sliced: list[int] = []
    cursor = 0
    for seg_len in segments:
        next_cursor = cursor + int(seg_len)
        overlap_start = max(cursor, start)
        overlap_end = min(next_cursor, end)
        overlap_len = overlap_end - overlap_start
        if overlap_len > 0:
            sliced.append(overlap_len)
        cursor = next_cursor
    return sliced


def _split_overlong_dialogue_once(
    row: dict,
    *,
    overlap_window: int,
) -> list[dict]:
    utterances = list(row["utterances"])
    segments = [int(seg_len) for seg_len in row["segments"]]
    total_len = len(utterances)
    if total_len <= 1:
        raise SystemExit(f"Cannot split dialogue {row['dial_id']} with only {total_len} utterance.")
    mid = total_len // 2
    first_start = 0
    first_end = min(mid + overlap_window, total_len)
    second_start = max(mid - overlap_window, 0)
    second_end = total_len
    if first_end <= first_start or second_end <= second_start:
        raise SystemExit(f"Invalid split range for dialogue {row['dial_id']}.")
    if first_end >= second_end and second_start <= first_start:
        raise SystemExit(f"Split does not reduce dialogue {row['dial_id']}.")
    first_segments = _slice_segments(segments, first_start, first_end)
    second_segments = _slice_segments(segments, second_start, second_end)
    return [
        {
            **row,
            "utterances": utterances[first_start:first_end],
            "segments": first_segments,
        },
        {
            **row,
            "utterances": utterances[second_start:second_end],
            "segments": second_segments,
        },
    ]


def _expand_overlong_dialogue(
    row: dict,
    *,
    max_utterances: int,
    overlap_window: int,
    max_parts: int,
) -> list[dict]:
    pending = [row]
    while True:
        overlong_idx = next(
            (idx for idx, item in enumerate(pending) if len(item["utterances"]) > max_utterances),
            None,
        )
        if overlong_idx is None:
            break
        if len(pending) >= max_parts:
            raise SystemExit(
                f"Pesudo dialogue {row['source_dial_id']} still exceeds max_utterances={max_utterances} "
                f"after expanding to {len(pending)} parts."
            )
        current = pending.pop(overlong_idx)
        split_rows = _split_overlong_dialogue_once(
            current,
            overlap_window=overlap_window,
        )
        pending[overlong_idx:overlong_idx] = split_rows
    total_parts = len(pending)
    expanded_rows: list[dict] = []
    for part_idx, item in enumerate(pending, start=1):
        expanded_row = {
            **item,
            "dial_id": f"{row['dial_id']}_part{part_idx}" if total_parts > 1 else row["dial_id"],
            "split_parent_dial_id": row["dial_id"],
            "split_part_index": part_idx,
            "split_total_parts": total_parts,
            "split_window": overlap_window,
        }
        if sum(expanded_row["segments"]) != len(expanded_row["utterances"]):
            raise SystemExit(
                f"Split dialogue {expanded_row['dial_id']} has mismatched segment lengths."
            )
        if len(expanded_row["utterances"]) > max_utterances:
            raise SystemExit(
                f"Split dialogue {expanded_row['dial_id']} still exceeds max_utterances={max_utterances}"
            )
        expanded_rows.append(expanded_row)
    return expanded_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/dataset/dialseg711.json")
    parser.add_argument("--artifact-dir", default="data/dataset/dialseg711_pesudo_100_artifacts")
    parser.add_argument("--label-file", default="data/dataset/dialseg711_pesudo_100_artifacts/pesudo_label_dialogue.jsonl")
    parser.add_argument("--openai-model", default="")
    parser.add_argument("--openai-retries", type=int, default=5)
    parser.add_argument("--openai-workers", type=int, default=4)
    parser.add_argument("--output", default="data/dataset/dialseg711_pesudo_100.json")
    parser.add_argument("--max-utterances", type=int, default=0)
    parser.add_argument("--min-segment-len", type=int, default=0)
    parser.add_argument("--max-segment-len", type=int, default=0)
    parser.add_argument("--split-window", type=int, default=2)
    parser.add_argument("--max-split-parts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.openai_model:
        args.openai_model = os.environ.get("OPENAI_MODEL") or os.environ.get("openai_model") or ""
    if not args.openai_model:
        raise SystemExit("Missing OpenAI model. Set `openai_model` in .env or pass --openai-model.")
    dataset_path = Path(args.dataset)
    data = load_dataset(str(dataset_path))
    split_map = split_dataset(data)
    train_dialogues = split_map["train"]
    valid_dialogues = split_map["valid"]
    test_dialogues = split_map["test"]

    real_segment_lengths = [int(seg_len) for dial in train_dialogues for seg_len in dial["segments"]]
    if not real_segment_lengths:
        raise SystemExit("No segment lengths found in train split.")
    positive_segment_lengths = [seg_len for seg_len in real_segment_lengths if seg_len > 0]
    if not positive_segment_lengths:
        raise SystemExit("No positive segment lengths found in train split.")
    real_mean = float(sum(real_segment_lengths) / len(real_segment_lengths))
    real_var = float(sum((seg_len - real_mean) ** 2 for seg_len in real_segment_lengths) / len(real_segment_lengths))
    real_std = math.sqrt(real_var)
    real_min = int(min(positive_segment_lengths))
    real_max = int(max(positive_segment_lengths))

    min_segment_len = int(args.min_segment_len) if args.min_segment_len > 0 else real_min
    max_segment_len = int(args.max_segment_len) if args.max_segment_len > 0 else real_max
    if min_segment_len <= 0 or max_segment_len <= 0 or min_segment_len > max_segment_len:
        raise SystemExit("Invalid segment length bounds.")
    max_utterances = int(args.max_utterances) if int(args.max_utterances) > 0 else _resolve_default_max_utterances(args.dataset, data)

    artifact_dir = Path(args.artifact_dir)
    label_path = Path(args.label_file)
    label_rows = _load_jsonl(label_path)
    if not label_rows:
        raise SystemExit(f"No pesudo label rows found in {label_path}")
    generated_path = artifact_dir / "pesudo_utterance_dialogue.jsonl"
    existing_generated_rows = _load_jsonl(generated_path)
    existing_by_id = {str(row["source_dial_id"]): row for row in existing_generated_rows}

    label_groups: dict[str, list[dict]] = {}
    for row in label_rows:
        label_groups.setdefault(str(row["source_dial_id"]), []).append(row)
    for rows in label_groups.values():
        rows.sort(key=lambda r: int(r["segments"][0]["source_segment_index"]) if r.get("segments") else 0)

    def _generate_one(idx: int, dial: dict) -> tuple[int, dict]:
        client = get_openai_client()
        local_rng = random.Random(args.seed + idx)
        pesudo_seg_specs: list[dict] = []
        for seg in dial["segments"]:
            target_len = _sample_segment_length(
                local_rng,
                mean=real_mean,
                std=real_std,
                min_len=min_segment_len,
                max_len=max_segment_len,
            )
            pesudo_seg_specs.append(
                {
                    "pesudo_label": seg["pesudo_label"],
                    "pesudo_summary": seg["pesudo_summary"],
                    "pesudo_seg_len": target_len,
                    "source_segment_index": seg["source_segment_index"],
                }
            )
        generated, raw_text, parse_error = generate_pesudo_dialogue_with_openai(
            client,
            model=args.openai_model,
            pesudo_seg_specs=pesudo_seg_specs,
            retries=args.openai_retries,
            debug_label=str(dial["source_dial_id"]),
        )
        if generated is None:
            print(
                f"Failed to generate dialogue for {dial['source_dial_id']}\n"
                f"Target segments: {[spec['pesudo_seg_len'] for spec in pesudo_seg_specs]}\n"
                f"Parse/validation error: {parse_error}\n"
                f"Raw output:\n{raw_text}",
                flush=True,
            )
            raise RuntimeError(
                f"Failed to generate dialogue for {dial['source_dial_id']}\n"
                f"Target segments: {[spec['pesudo_seg_len'] for spec in pesudo_seg_specs]}\n"
                f"Parse/validation error: {parse_error}\n"
                f"Raw output:\n{raw_text}"
            )
        utterances, segments = generated
        item = {
            "source_dial_id": dial["source_dial_id"],
            "pesudo_seg_specs": pesudo_seg_specs,
            "target_pesudo_seg": [spec["pesudo_seg_len"] for spec in pesudo_seg_specs],
            "utterances": utterances,
            "segments": segments,
        }
        return idx, item

    generated_rows: list[dict | None] = [existing_by_id.get(str(dial["source_dial_id"])) for dial in label_rows]
    pending_dialogues = [
        (idx, dial)
        for idx, dial in enumerate(label_rows)
        if str(dial["source_dial_id"]) not in existing_by_id
    ]
    if pending_dialogues:
        with ThreadPoolExecutor(max_workers=max(int(args.openai_workers), 1)) as executor:
            futures = [executor.submit(_generate_one, idx, dial) for idx, dial in pending_dialogues]
            for future in tqdm(as_completed(futures), total=len(futures), desc="pesudo utterance", unit="dlg"):
                idx, item = future.result()
                generated_rows[idx] = item
                completed_rows = [row for row in generated_rows if row is not None]
                save_jsonl(generated_path, completed_rows)

    final_generated_rows: list[dict] = []
    for item in generated_rows:
        if item is None:
            raise SystemExit("OpenAI generation produced an unexpected empty result.")
        final_generated_rows.append(item)

    save_jsonl(generated_path, final_generated_rows)

    pesudo_train: list[dict] = []
    split_dialogue_count = 0
    for row in tqdm(final_generated_rows, desc="build pesudo", unit="dlg"):
        if sum(row["segments"]) != len(row["utterances"]):
            raise SystemExit(f"Segment lengths do not match utterances for {row['source_dial_id']}")
        base_item = {
            "dial_id": f"{row['source_dial_id']}_pesudo",
            "utterances": row["utterances"],
            "segments": row["segments"],
            "set": "train",
            "source_dial_id": row["source_dial_id"],
        }
        split_items = _expand_overlong_dialogue(
            base_item,
            max_utterances=max_utterances,
            overlap_window=max(int(args.split_window), 0),
            max_parts=max(int(args.max_split_parts), 1),
        )
        if len(split_items) > 1:
            split_dialogue_count += 1
        pesudo_train.extend(split_items)

    final_dataset = pesudo_train + [{**item, "set": "valid"} for item in valid_dialogues] + [{**item, "set": "test"} for item in test_dialogues]
    output = Path(args.output)
    save_json(output, final_dataset)
    save_json(
        artifact_dir / "pesudo_utterance_manifest.json",
        {
            "label_file": str(label_path),
            "label_dialogue_count": len(label_rows),
            "generated_dialogue_count": len(final_generated_rows),
            "openai_retries": args.openai_retries,
            "openai_workers": args.openai_workers,
            "resumed_dialogue_count": len(existing_by_id),
            "pending_dialogue_count": len(pending_dialogues),
            "seed": args.seed,
            "pesudo_train_count": len(pesudo_train),
            "split_dialogue_count": split_dialogue_count,
            "split_window": int(args.split_window),
            "max_split_parts": int(args.max_split_parts),
            "valid_count": len(valid_dialogues),
            "test_count": len(test_dialogues),
            "real_segment_len_mean": real_mean,
            "real_segment_len_std": real_std,
            "real_segment_len_min": real_min,
            "real_segment_len_max": real_max,
            "sampling_segment_len_min": min_segment_len,
            "sampling_segment_len_max": max_segment_len,
            "max_utterances": max_utterances,
            "output": str(output),
        },
    )
    print(f"Wrote pesudo train to {output}")


if __name__ == "__main__":
    main()
