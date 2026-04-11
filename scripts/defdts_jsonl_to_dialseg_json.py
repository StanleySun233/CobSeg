import argparse
import json
from pathlib import Path

NEWLINE = "[NEWLINE]"
BOUNDARY = "[BOUNDARY]"


def dialogue_to_utterances_and_segments(dialogue: str):
    utterances = []
    segments = []
    seg_count = 0
    for raw in dialogue.split(NEWLINE):
        part = raw.strip()
        if not part:
            continue
        if part == BOUNDARY:
            if seg_count > 0:
                segments.append(seg_count)
            seg_count = 0
            continue
        if part.startswith("user: "):
            utterances.append(part[6:])
            seg_count += 1
            continue
        if part.startswith("agent: "):
            utterances.append(part[7:])
            seg_count += 1
            continue
    if seg_count > 0:
        segments.append(seg_count)
    return utterances, segments


def convert_jsonl(path: Path, set_name: str):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            utts, segs = dialogue_to_utterances_and_segments(obj["dialogue"])
            rows.append(
                {
                    "utterances": utts,
                    "segments": segs,
                    "set": set_name,
                }
            )
    return rows


def merge_splits(rows_by_split):
    merged = []
    dial_id = 0
    for rows in rows_by_split:
        for r in rows:
            merged.append({"dial_id": dial_id, **r})
            dial_id += 1
    return merged


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "comp"
        / "Def-DTS"
        / "data"
        / "DTS_session_datasets",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "data" / "dataset",
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = [
        (
            "superseg.json",
            [
                ("superseg_train.jsonl", "train"),
                ("superseg_validation.jsonl", "valid"),
                ("superseg_test.jsonl", "test"),
            ],
        ),
        (
            "tiage.json",
            [
                ("tiage_train.jsonl", "train"),
                ("tiage_validation.jsonl", "valid"),
                ("tiage_test.jsonl", "test"),
            ],
        ),
    ]
    for out_name, split_specs in datasets:
        parts = []
        for src_name, set_name in split_specs:
            src = args.input_dir / src_name
            if not src.is_file():
                parts.append([])
                continue
            parts.append(convert_jsonl(src, set_name))
        data = merge_splits(parts)
        out_path = args.output_dir / out_name
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
