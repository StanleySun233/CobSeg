"""
Convert ~/maritime/dts/data/dataset/*.json  →  T5-NLG TSV format

Each JSON file has the unified schema:
  { "dial_id": int, "utterances": [str, ...], "segments": [int, ...], "set": "train|valid|test" }

Target TSV format (matches TIAGE's ws_train_instances_all_with_eos.tsv):
  <context>  TAB  <response>
where <context> is all preceding utterances joined by " <eos> ", ending with " <eos>".

Output layout (one sub-dir per source dataset):
  data/t5_nlg/<dataset>/train.tsv
  data/t5_nlg/<dataset>/valid.tsv
  data/t5_nlg/<dataset>/test.tsv

Usage
-----
  # single dataset
  python scripts/convert_to_t5_nlg.py --datasets vhf

  # all datasets
  python scripts/convert_to_t5_nlg.py --datasets all

  # all datasets, minimum context window of 2 turns
  python scripts/convert_to_t5_nlg.py --datasets all --min_context 2
"""

import argparse
import json
import os

# ── config ──────────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), "../data/dataset")
OUTPUT_ROOT = os.path.join(os.path.dirname(__file__), "../data/t5_nlg")
ALL_DATASETS = ["tiage", "dialseg711", "doc2dial", "vhf", "superseg"]
SPLITS = ["train", "valid", "test"]
SEP = " <eos> "   # turn separator, matches TIAGE training convention


# ── core conversion ──────────────────────────────────────────────────────────

def dialog_to_nlg_rows(utterances: list[str], min_context: int = 1) -> list[tuple[str, str]]:
    """
    Convert a single dialog (list of utterances) into (context, response) pairs.

    Parameters
    ----------
    utterances  : ordered list of utterance strings for one dialog
    min_context : minimum number of preceding turns required before we emit a row
                  (default 1: every turn starting from index 1 is used)

    Returns
    -------
    List of (context_str, response_str) tuples ready to write as TSV rows.
    """
    rows = []
    for i in range(min_context, len(utterances)):
        context = SEP.join(utterances[:i]) + SEP   # trailing <eos> matches original format
        response = utterances[i]
        rows.append((context, response))
    return rows


def convert_dataset(name: str, min_context: int = 1) -> dict[str, int]:
    """
    Load one JSON dataset file, convert to NLG TSV, write per-split files.

    Returns a dict with row counts per split, e.g. {"train": 4200, "valid": 800, "test": 750}
    """
    src = os.path.join(DATASET_DIR, f"{name}.json")
    with open(src, encoding="utf-8") as f:
        dialogs = json.load(f)

    # Bucket dialogs by split
    buckets: dict[str, list] = {s: [] for s in SPLITS}
    for d in dialogs:
        split = d.get("set", "train")
        if split in buckets:
            buckets[split].append(d)

    out_dir = os.path.join(OUTPUT_ROOT, name)
    os.makedirs(out_dir, exist_ok=True)

    counts: dict[str, int] = {}
    for split, split_dialogs in buckets.items():
        out_path = os.path.join(out_dir, f"{split}.tsv")
        n_rows = 0
        with open(out_path, "w", encoding="utf-8") as fout:
            for d in split_dialogs:
                utts = d["utterances"]
                for context, response in dialog_to_nlg_rows(utts, min_context):
                    fout.write(f"{context}\t{response}\n")
                    n_rows += 1
        counts[split] = n_rows
        print(f"  [{name}] {split:5s}: {len(split_dialogs):5d} dialogs → {n_rows:6d} rows  →  {out_path}")

    return counts


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert JSON dialogs to T5-NLG TSV format")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["all"],
        help='Dataset names to convert, or "all" to process every dataset. '
             f'Available: {ALL_DATASETS}',
    )
    parser.add_argument(
        "--min_context",
        type=int,
        default=1,
        help="Minimum number of preceding turns required to emit an NLG row (default: 1)",
    )
    args = parser.parse_args()

    targets = ALL_DATASETS if "all" in args.datasets else args.datasets
    for name in targets:
        src = os.path.join(DATASET_DIR, f"{name}.json")
        if not os.path.exists(src):
            print(f"[SKIP] {name}.json not found at {src}")
            continue
        print(f"\nConverting: {name}")
        convert_dataset(name, min_context=args.min_context)

    print(f"\nDone. Output written to: {os.path.abspath(OUTPUT_ROOT)}/")


if __name__ == "__main__":
    main()
