"""
Zero-shot T5-base topic-shift detection evaluation on five datasets.

Task: For every consecutive utterance pair in a dialogue, T5 generates
"positive" (topic shift) or "negative" (no shift). These binary boundary
predictions are then evaluated with Pk, WD, F1, and Score using the same
metric functions as the main DTS codebase (utils/metrics.py).

Input:  data/dataset/{tiage,dialseg711,doc2dial,vhf,superseg}.json
Output: data/t5_eval/results.json   (machine-readable)
        data/t5_eval/results.tsv    (copy-paste for paper)

Prompt format fed to T5 (matches TIAGE classifier convention):
  "context: <utt_0> <eos> ... <utt_i-1> <eos> response: <utt_i>"
  → T5 generates "positive" or "negative"

Because this is ZERO-SHOT, T5-base will likely output arbitrary tokens.
We treat any output that starts with "pos" as a shift boundary (1) and
everything else as no-shift (0).

Usage
-----
  python scripts/eval_t5_ts.py                          # all datasets
  python scripts/eval_t5_ts.py --datasets vhf tiage     # subset
  python scripts/eval_t5_ts.py --max_samples 100        # smoke-test
  python scripts/eval_t5_ts.py --model_name_or_path t5-large
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

# ── reuse project metrics ────────────────────────────────────────────────────
PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
from utils.dts_utils import evaluate_all, segments_to_boundaries

# ── constants ────────────────────────────────────────────────────────────────
ALL_DATASETS = ["tiage", "dialseg711", "doc2dial", "vhf", "superseg"]
DATASET_DIR  = os.path.join(PROJECT_DIR, "data", "dataset")
OUTPUT_DIR   = os.path.join(PROJECT_DIR, "data", "t5_eval")

METRIC_COLS = ["PK", "WD", "F1", "Precision", "Recall", "Score"]


# ── data ─────────────────────────────────────────────────────────────────────

def load_test_dialogues(name: str) -> list[dict]:
    path = os.path.join(DATASET_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return [d for d in data if d.get("set") == "test"]


class BoundaryDataset(Dataset):
    """
    One sample = one (context, current_utt) pair → predict shift or not.

    context  : utt[0] <eos> utt[1] <eos> ... utt[i-1] <eos>
    current  : utt[i]
    prompt   : "context: <context> response: <current>"
    label    : 1 if utt[i-1] is a segment boundary, else 0
    """

    def __init__(self, dialogues: list[dict], tokenizer, max_input_len: int, max_samples: int = 0):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.prompts: list[str] = []
        self.labels: list[int] = []
        # keep track of dial/position so we can reassemble per-dialogue lists
        self.dial_ids: list[int] = []
        self.positions: list[int] = []   # index i within utterances

        for d in dialogues:
            utts = d["utterances"]
            boundaries = segments_to_boundaries(d["segments"])
            # boundary[i] = 1 means there is a topic shift AFTER utt[i]
            # For prediction: given context=[0..i-1] + current=utt[i],
            # the label is boundaries[i-1] (was the previous utt a boundary?)
            # Edge case: i=1 → no previous boundary, label=0
            for i in range(1, len(utts)):
                label = boundaries[i - 1] if i - 1 < len(boundaries) else 0
                context = " <eos> ".join(utts[:i]) + " <eos>"
                prompt  = f"context: {context} response: {utts[i]} label: positive or negative?"
                self.prompts.append(prompt)
                self.labels.append(label)
                self.dial_ids.append(d["dial_id"])
                self.positions.append(i)

                if max_samples and len(self.prompts) >= max_samples:
                    break
            if max_samples and len(self.prompts) >= max_samples:
                break

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.prompts[idx],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label":          self.labels[idx],
            "dial_id":        self.dial_ids[idx],
            "position":       self.positions[idx],
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         [b["label"]    for b in batch],
        "dial_ids":       [b["dial_id"]  for b in batch],
        "positions":      [b["position"] for b in batch],
    }


# ── generation & decoding ────────────────────────────────────────────────────

def decode_prediction(text: str) -> int:
    """Map T5 output text → 0 or 1.  "positive..." → 1, anything else → 0."""
    return 1 if text.strip().lower().startswith("pos") else 0


def run_generation(
    model,
    tokenizer,
    dataset: BoundaryDataset,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> tuple[list[int], list[int], list, list]:
    """Returns (hyp_flat, ref_flat, dial_ids_flat, positions_flat)."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    hyps, refs, dids, poss = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  generating", leave=False):
            outs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
            preds = [decode_prediction(t) for t in decoded]
            hyps.extend(preds)
            refs.extend(batch["labels"])
            dids.extend(batch["dial_ids"])
            poss.extend(batch["positions"])

    return hyps, refs, dids, poss


# ── reassemble per-dialogue boundary lists ───────────────────────────────────

def reassemble(
    dialogues: list[dict],
    hyps: list[int],
    refs: list[int],
    dial_ids: list,
    positions: list[int],
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Regroup flat prediction/label lists back into per-dialogue boundary lists.
    The boundary list has length = len(utterances) (one entry per utterance).
      - boundary[i] = predicted/true label for the transition AFTER utt[i]
      - position i in our dataset = "given context[0..i-1], predict shift after utt[i-1]"
        so the boundary slot we fill is position-1.
    """
    # Build lookup: dial_id → (num_utterances, segments)
    dial_info = {str(d["dial_id"]): d for d in dialogues}

    # Collect predictions by dialogue
    from collections import defaultdict
    pred_by_dial: dict[str, dict[int, int]] = defaultdict(dict)
    true_by_dial: dict[str, dict[int, int]] = defaultdict(dict)

    for h, r, did, pos in zip(hyps, refs, dial_ids, positions):
        key = str(did)
        slot = pos - 1          # boundary slot: after utt[pos-1]
        pred_by_dial[key][slot] = h
        true_by_dial[key][slot] = r

    all_preds, all_labels = [], []
    for d in dialogues:
        key = str(d["dial_id"])
        n = len(d["utterances"])
        true_bounds = segments_to_boundaries(d["segments"])

        pred_bounds = []
        for i in range(n):
            pred_bounds.append(pred_by_dial[key].get(i, 0))

        all_preds.append(pred_bounds)
        all_labels.append(true_bounds)

    return all_preds, all_labels


# ── table helpers ─────────────────────────────────────────────────────────────

def print_table(all_results: dict[str, dict]):
    col_w = 10
    header = f"{'Dataset':<14}" + "".join(f"{m:>{col_w}}" for m in METRIC_COLS)
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for name, metrics in all_results.items():
        row = f"{name:<14}" + "".join(f"{metrics[m]:>{col_w}.4f}" for m in METRIC_COLS)
        print(row)
    print(sep)


def write_results(all_results: dict, model_name: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    json_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump({"model": model_name, "results": all_results}, f, indent=2)
    print(f"\nSaved → {json_path}")

    tsv_path = os.path.join(OUTPUT_DIR, "results.tsv")
    with open(tsv_path, "w") as f:
        f.write("dataset\t" + "\t".join(METRIC_COLS) + "\n")
        for name, m in all_results.items():
            f.write(name + "\t" + "\t".join(f"{m[c]:.4f}" for c in METRIC_COLS) + "\n")
    print(f"Saved → {tsv_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="t5-base")
    p.add_argument("--datasets",     nargs="+", default=["all"])
    p.add_argument("--batch_size",   type=int,  default=64)
    p.add_argument("--max_input_len",type=int,  default=512)
    p.add_argument("--max_new_tokens",type=int, default=4,
                   help="T5 only needs to generate 'positive' or 'negative'")
    p.add_argument("--num_beams",    type=int,  default=2)
    p.add_argument("--max_samples",  type=int,  default=0,
                   help="Cap total pairs per dataset (0 = all)")
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    targets = ALL_DATASETS if "all" in args.datasets else args.datasets

    print(f"\nLoading model : {args.model_name_or_path}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model     = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)
    print(f"Params        : {sum(p.numel() for p in model.parameters())/1e6:.0f}M\n")

    all_results: dict[str, dict] = {}

    for name in targets:
        json_path = os.path.join(DATASET_DIR, f"{name}.json")
        if not os.path.exists(json_path):
            print(f"[SKIP] {name}: file not found at {json_path}")
            continue

        print(f"── {name} ──")
        t0 = time.time()

        dialogues = load_test_dialogues(name)
        print(f"  {len(dialogues)} test dialogues")

        dataset = BoundaryDataset(
            dialogues, tokenizer,
            max_input_len=args.max_input_len,
            max_samples=args.max_samples,
        )
        print(f"  {len(dataset)} utterance pairs")

        hyps, refs, dial_ids, positions = run_generation(
            model, tokenizer, dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            device=device,
        )

        all_preds, all_labels = reassemble(dialogues, hyps, refs, dial_ids, positions)
        metrics = evaluate_all(all_preds, all_labels)
        all_results[name] = metrics

        elapsed = time.time() - t0
        print(
            f"  PK={metrics['PK']:.4f}  WD={metrics['WD']:.4f}  "
            f"F1={metrics['F1']:.4f}  Score={metrics['Score']:.4f}  [{elapsed:.0f}s]\n"
        )

    print_table(all_results)
    write_results(all_results, args.model_name_or_path)


if __name__ == "__main__":
    main()
