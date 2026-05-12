from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env", override=False)

from utils.pesudo_common import FlanT5Summarizer, filter_pesudo_seg_rows, save_json, save_jsonl, build_pesudo_summary_rows


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", default="data/dataset/dialseg711_pesudo_50_artifacts")
    parser.add_argument("--summary-model", default="google-t5/t5-large")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    pesudo_segments = _load_jsonl(artifact_dir / "pesudo_seg_samples.jsonl")
    if not pesudo_segments:
        raise SystemExit(f"No pesudo seg samples found in {artifact_dir / 'pesudo_seg_samples.jsonl'}")

    pesudo_seg_manifest_path = artifact_dir / "pesudo_seg_manifest.json"
    avg_turn = None
    if pesudo_seg_manifest_path.exists():
        pesudo_seg_manifest = json.loads(pesudo_seg_manifest_path.read_text(encoding="utf-8"))
        avg_turn = int(pesudo_seg_manifest.get("avg_turn", 0)) or None
    if avg_turn is None:
        all_segment_lengths = [len(seg["utterances"]) for seg in pesudo_segments if seg["utterances"]]
        avg_turn = max(int(np.floor(np.mean(all_segment_lengths))) if all_segment_lengths else 1, 1)

    filtered_segments = filter_pesudo_seg_rows(
        pesudo_segments,
        min_segment_len=int(avg_turn),
        max_segment_len=10,
    )
    if not filtered_segments:
        raise SystemExit(
            "No pesudo seg samples satisfy the filter "
            f"(segment_len >= {int(avg_turn)} and segment_len < 10)."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summarizer = FlanT5Summarizer(args.summary_model, device)

    summary_rows = build_pesudo_summary_rows(
        summarizer,
        filtered_segments,
        desc="pesudo summary",
        batch_size=args.batch_size,
    )
    pesudo_summary_path = artifact_dir / "pesudo_summary.jsonl"
    save_jsonl(pesudo_summary_path, summary_rows)
    save_json(
        artifact_dir / "pesudo_summary_manifest.json",
        {
            "avg_turn": avg_turn,
            "min_segment_len": int(avg_turn),
            "max_segment_len": 10,
            "filtered_dialogue_count": len({row["source_dial_id"] for row in filtered_segments}),
            "filtered_segment_count": len(filtered_segments),
            "summary_count": len(summary_rows),
            "summary_model": args.summary_model,
            "batch_size": args.batch_size,
            "device": str(device),
            "output": str(pesudo_summary_path),
            "source_pesudo_seg_manifest": str(pesudo_seg_manifest_path),
        },
    )
    if args.output:
        save_json(
            Path(args.output),
            {
                "summary_count": len(summary_rows),
                "output": str(pesudo_summary_path),
            },
        )
    print(f"Wrote pesudo summaries to {pesudo_summary_path}")


if __name__ == "__main__":
    main()
