from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
from collections import Counter

import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)

from utils.texttiling_baseline import segment_texts
from utils.pesudo_common import (
    generate_pesudo_seg_labels_with_openai,
    get_openai_client,
    save_json,
    save_jsonl,
)


def _load_json(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list at {path}")
    return payload


def _resolve_pesudo_seg_path(artifact_dir: Path) -> Path:
    generic_path = artifact_dir / "pesudo_seg_sampled_train.json"
    if generic_path.exists():
        return generic_path
    candidates = sorted(artifact_dir.glob("pesudo_seg_sampled_train_*.json"))
    if not candidates:
        raise SystemExit(f"No pesudo seg train file found in {artifact_dir}")
    return candidates[0]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="data/dataset/dialseg711_pesudo_100_artifacts")
    parser.add_argument("--openai-model", default="")
    parser.add_argument("--openai-retries", type=int, default=5)
    parser.add_argument("--openai-workers", type=int, default=4)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if not args.openai_model:
        args.openai_model = os.environ.get("OPENAI_MODEL") or os.environ.get("openai_model") or ""
    if not args.openai_model:
        raise SystemExit("Missing OpenAI model. Set `openai_model` in .env or pass --openai-model.")

    artifact_dir = Path(args.artifact_dir)
    sampled_path = _resolve_pesudo_seg_path(artifact_dir)
    dialogues = _load_json(sampled_path)
    if not dialogues:
        raise SystemExit(f"No dialogues found in {sampled_path}")
    pesudo_label_path = Path(args.output) if args.output else artifact_dir / "pesudo_label_dialogue.jsonl"
    existing_rows = _load_jsonl(pesudo_label_path)
    existing_by_id = {str(row["source_dial_id"]): row for row in existing_rows}
    pending_dialogues = [
        (idx, dial)
        for idx, dial in enumerate(dialogues)
        if str(dial["dial_id"]) not in existing_by_id
    ]

    def _summarize_one(idx: int, dial: dict) -> tuple[int, dict]:
        client = get_openai_client()
        segment_texts_list = segment_texts(dial["utterances"], dial["segments"])
        payload, raw_text, parse_error = generate_pesudo_seg_labels_with_openai(
            client,
            model=args.openai_model,
            segments=segment_texts_list,
            retries=args.openai_retries,
            debug_label=str(dial["dial_id"]),
        )
        if payload is None:
            print(
                f"Failed to generate pesudo labels for {dial['dial_id']}\n"
                f"Parse/validation error: {parse_error}\n"
                f"Raw output:\n{raw_text}"
            , flush=True)
            raise SystemExit(1)
        segments_out: list[dict] = []
        for seg_idx, item in enumerate(payload):
            segments_out.append(
                {
                    "source_segment_index": seg_idx,
                    "segment_len": len(segment_texts_list[seg_idx]),
                    "pesudo_label": item["pesudo_label"],
                    "pesudo_summary": item["pesudo_summary"],
                }
            )
        return idx, {
            "source_dial_id": dial["dial_id"],
            "utterances": dial["utterances"],
            "segments": segments_out,
        }

    pesudo_label_results: list[dict | None] = [existing_by_id.get(str(dial["dial_id"])) for dial in dialogues]
    if pending_dialogues:
        with ThreadPoolExecutor(max_workers=max(int(args.openai_workers), 1)) as executor:
            futures = [executor.submit(_summarize_one, idx, dial) for idx, dial in pending_dialogues]
            for future in tqdm(as_completed(futures), total=len(futures), desc="pesudo label", unit="dlg"):
                idx, item = future.result()
                pesudo_label_results[idx] = item
                completed_rows = [row for row in pesudo_label_results if row is not None]
                save_jsonl(pesudo_label_path, completed_rows)

    pesudo_label_rows: list[dict] = []
    for item in pesudo_label_results:
        if item is None:
            raise SystemExit("Pesudo label generation produced an unexpected empty result.")
        pesudo_label_rows.append(item)

    save_jsonl(pesudo_label_path, pesudo_label_rows)
    label_counts = Counter(seg["pesudo_label"] for row in pesudo_label_rows for seg in row["segments"])
    save_json(
        artifact_dir / "pesudo_label_manifest.json",
        {
            "dialogue_count": len(pesudo_label_rows),
            "segment_count": sum(len(row["segments"]) for row in pesudo_label_rows),
            "pesudo_label_counts": dict(label_counts),
            "openai_model": args.openai_model,
            "openai_retries": args.openai_retries,
            "openai_workers": args.openai_workers,
            "resumed_dialogue_count": len(existing_by_id),
            "pending_dialogue_count": len(pending_dialogues),
            "output": str(pesudo_label_path),
        },
    )
    print(f"Wrote pesudo labels to {pesudo_label_path}")


if __name__ == "__main__":
    main()
