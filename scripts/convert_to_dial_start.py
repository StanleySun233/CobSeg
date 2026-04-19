#!/usr/bin/env python3

import argparse
import json
import re
from pathlib import Path


def dataset_mapping(repo_root: Path) -> dict[str, Path]:
    return {
        "vhf": repo_root / "data" / "dataset" / "vhf.json",
        "dialseg711": repo_root / "data" / "dataset" / "dialseg711.json",
        "doc2dial": repo_root / "data" / "dataset" / "doc2dial.json",
        "tiage": repo_root / "data" / "dataset" / "tiage.json",
        "superseg": repo_root / "data" / "dataset" / "superseg.json",
    }


def resolve_dataset_path(dataset_name_or_path: str, repo_root: Path) -> Path:
    if dataset_name_or_path.endswith(".json"):
        return Path(dataset_name_or_path).resolve()

    mapping = dataset_mapping(repo_root)
    return mapping[dataset_name_or_path]


def sanitize_filename(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value.strip())
    return value or "dialogue"


def validate_dialogue(item: dict, idx: int) -> None:
    required = ("utterances", "segments")
    missing = [key for key in required if key not in item]
    if missing:
        raise ValueError(f"dialogue #{idx} missing keys: {missing}")

    utterances = item["utterances"]
    segments = item["segments"]
    if not isinstance(utterances, list) or not isinstance(segments, list):
        raise ValueError(f"dialogue #{idx} has invalid utterances/segments type")
    if sum(int(x) for x in segments) != len(utterances):
        raise ValueError(
            f"dialogue #{idx} has mismatched lengths: sum(segments)={sum(int(x) for x in segments)} "
            f"but len(utterances)={len(utterances)}"
        )


def dialogue_to_lines(item: dict, boundary_marker: str) -> list[str]:
    utterances = [str(u).replace("\n", " ").strip() for u in item["utterances"]]
    segments = [int(x) for x in item["segments"]]

    lines: list[str] = []
    cursor = 0
    for seg_idx, seg_len in enumerate(segments):
        lines.extend(utterances[cursor: cursor + seg_len])
        cursor += seg_len
        if seg_idx != len(segments) - 1:
            lines.append(boundary_marker)
    return lines


def convert_one_dataset(
    input_path: Path,
    output_root: Path,
    output_name: str,
    split_mode: str,
    boundary_marker: str,
) -> dict:
    data = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise SystemExit(f"Expected a list of dialogues in {input_path}")

    dataset_out_dir = output_root / output_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    counts_by_set: dict[str, int] = {}
    written = 0

    for idx, item in enumerate(data):
        validate_dialogue(item, idx)

        split_name = str(item.get("set", "unknown"))
        counts_by_set[split_name] = counts_by_set.get(split_name, 0) + 1

        if split_mode == "by_set":
            target_dir = dataset_out_dir / split_name
            target_dir.mkdir(parents=True, exist_ok=True)
        else:
            target_dir = dataset_out_dir

        dial_id = sanitize_filename(str(item.get("dial_id", idx)))
        file_path = target_dir / f"{dial_id}.txt"
        if file_path.exists():
            file_path = target_dir / f"{dial_id}_{idx}.txt"

        lines = dialogue_to_lines(item, boundary_marker)
        file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        written += 1

    return {
        "input_json": str(input_path),
        "output_dir": str(dataset_out_dir),
        "dialogues_written": written,
        "split_mode": split_mode,
        "counts_by_set": counts_by_set,
        "boundary_marker": boundary_marker,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert base dialogue JSON into DialSTART text-file format."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset name (vhf|dialseg711|doc2dial|tiage|superseg|all) or path to a JSON file.",
    )
    parser.add_argument(
        "--output_root",
        default="comp/dial-start/data",
        help="Root directory where the converted dataset directory will be created.",
    )
    parser.add_argument(
        "--output_name",
        default="",
        help="Output dataset directory name. Defaults to the input JSON stem.",
    )
    parser.add_argument(
        "--split_mode",
        choices=("flat", "by_set"),
        default="flat",
        help="flat: write all dialogues into one directory for DialSTART compatibility; "
             "by_set: create train/valid/test subdirectories.",
    )
    parser.add_argument(
        "--boundary_marker",
        default="=======",
        help="Segment boundary line inserted between segments.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    output_root = (repo_root / args.output_root).resolve()
    if args.dataset == "all":
        all_summaries = []
        for dataset_name, input_path in dataset_mapping(repo_root).items():
            if not input_path.exists():
                raise SystemExit(f"Input JSON not found: {input_path}")
            summary = convert_one_dataset(
                input_path=input_path,
                output_root=output_root,
                output_name=dataset_name,
                split_mode=args.split_mode,
                boundary_marker=args.boundary_marker,
            )
            all_summaries.append(summary)
            print(f"Input JSON: {summary['input_json']}")
            print(f"Output dir: {summary['output_dir']}")
            print(f"Dialogues written: {summary['dialogues_written']}")
            print(f"Split counts: {summary['counts_by_set']}")
            print()
        return

    input_path = resolve_dataset_path(args.dataset, repo_root)
    if not input_path.exists():
        raise SystemExit(f"Input JSON not found: {input_path}")

    output_name = args.output_name or input_path.stem
    summary = convert_one_dataset(
        input_path=input_path,
        output_root=output_root,
        output_name=output_name,
        split_mode=args.split_mode,
        boundary_marker=args.boundary_marker,
    )

    print(f"Input JSON: {summary['input_json']}")
    print(f"Output dir: {summary['output_dir']}")
    print(f"Dialogues written: {summary['dialogues_written']}")
    print(f"Split counts: {summary['counts_by_set']}")


if __name__ == "__main__":
    main()
