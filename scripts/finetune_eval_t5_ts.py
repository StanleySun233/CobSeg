"""
Fine-tune T5-base for topic-shift detection, then evaluate on all five datasets.

For each dataset:
  1. Fine-tune t5-base on the train split (seq2seq: input=prompt, target="positive"/"negative")
  2. Evaluate on the test split with Pk, WD, F1, Precision, Recall, Score
  3. Save per-dataset checkpoint to data/t5_finetune/<dataset>/checkpoint/
  4. Write results to data/t5_finetune/results.json and results.tsv

Prompt format:
  "context: utt1 <eos> ... utt_i-1 <eos> response: utt_i label: positive or negative?"
Target:
  "positive"  (topic shift boundary)  /  "negative"  (no boundary)

Usage
-----
  python scripts/finetune_eval_t5_ts.py                        # all datasets, 3 epochs
  python scripts/finetune_eval_t5_ts.py --datasets vhf tiage
  python scripts/finetune_eval_t5_ts.py --epochs 5
  python scripts/finetune_eval_t5_ts.py --max_samples 300      # smoke-test
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer, get_linear_schedule_with_warmup

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_DIR)
from utils.dts_utils import evaluate_all, segments_to_boundaries

# ── constants ────────────────────────────────────────────────────────────────
ALL_DATASETS = ["tiage", "dialseg711", "doc2dial", "vhf", "superseg"]
DATASET_DIR  = os.path.join(PROJECT_DIR, "data", "dataset")
OUTPUT_ROOT  = os.path.join(PROJECT_DIR, "data", "t5_finetune")
METRIC_COLS  = ["PK", "WD", "F1", "Precision", "Recall", "Score"]

POS_LABEL = "positive"
NEG_LABEL = "negative"


# ── dataset ──────────────────────────────────────────────────────────────────

def make_prompt(context: str, response: str) -> str:
    return f"context: {context} response: {response} label: positive or negative?"


class BoundaryDataset(Dataset):
    """Utterance-pair dataset for seq2seq training/evaluation."""

    def __init__(
        self,
        dialogues: list[dict],
        tokenizer,
        max_input_len: int,
        split: str = "train",       # "train" | "valid" | "test"
        max_samples: int = 0,       # 0 = all; >0 = full dataset (ignored for train when few_shot_k set)
        few_shot_k: int = 0,        # if >0 (train only): sample k pos + k neg pairs
    ):
        self.tokenizer     = tokenizer
        self.max_input_len = max_input_len
        self.prompts:   list[str] = []
        self.targets:   list[str] = []   # "positive" or "negative"
        self.labels:    list[int] = []   # 1 or 0  (for metrics)
        self.dial_ids:  list     = []
        self.positions: list[int] = []

        all_pos, all_neg = [], []   # (prompt, target, label, dial_id, position)

        for d in [x for x in dialogues if x.get("set") == split]:
            utts       = d["utterances"]
            boundaries = segments_to_boundaries(d["segments"])

            for i in range(1, len(utts)):
                label  = boundaries[i - 1] if i - 1 < len(boundaries) else 0
                context = " <eos> ".join(utts[:i]) + " <eos>"
                prompt  = make_prompt(context, utts[i])
                target  = POS_LABEL if label == 1 else NEG_LABEL
                entry   = (prompt, target, label, d["dial_id"], i)

                if label == 1:
                    all_pos.append(entry)
                else:
                    all_neg.append(entry)

        if few_shot_k > 0 and split == "train":
            # sample k positive + k negative pairs (without replacement if possible)
            import random
            sampled_pos = random.sample(all_pos, min(few_shot_k, len(all_pos)))
            sampled_neg = random.sample(all_neg, min(few_shot_k, len(all_neg)))
            entries = sampled_pos + sampled_neg
            random.shuffle(entries)
        else:
            entries = all_pos + all_neg
            # restore original order by dial_id / position
            entries.sort(key=lambda e: (e[3], e[4]))

        for prompt, target, label, dial_id, position in entries:
            self.prompts.append(prompt)
            self.targets.append(target)
            self.labels.append(label)
            self.dial_ids.append(dial_id)
            self.positions.append(position)

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
        tgt = self.tokenizer(
            self.targets[idx],
            max_length=8,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        labels = tgt["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100   # ignore pad in loss

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels":         labels,
            "true_label":     self.labels[idx],
            "dial_id":        self.dial_ids[idx],
            "position":       self.positions[idx],
        }


def train_collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels":         torch.stack([b["labels"]         for b in batch]),
    }


def eval_collate(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "true_labels":    [b["true_label"] for b in batch],
        "dial_ids":       [b["dial_id"]    for b in batch],
        "positions":      [b["position"]   for b in batch],
    }


# ── training ──────────────────────────────────────────────────────────────────

def train(model, train_loader, optimizer, scheduler, device, epoch, epochs):
    model.train()
    total_loss = 0.0
    pbar = tqdm(train_loader, desc=f"  train {epoch}/{epochs}", leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        out = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            labels=batch["labels"].to(device),
        )
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(len(train_loader), 1)


# ── evaluation (generate then score) ─────────────────────────────────────────

def decode_pred(text: str) -> int:
    return 1 if text.strip().lower().startswith("pos") else 0


def predict(model, loader, tokenizer, device):
    """Returns (hyps, true_labels, dial_ids, positions) — all flat lists."""
    model.eval()
    hyps, refs, dids, poss = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="  eval", leave=False):
            outs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=4,
                num_beams=2,
            )
            decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
            hyps.extend(decode_pred(t) for t in decoded)
            refs.extend(batch["true_labels"])
            dids.extend(batch["dial_ids"])
            poss.extend(batch["positions"])
    return hyps, refs, dids, poss


def reassemble(dialogues, hyps, refs, dial_ids, positions):
    from collections import defaultdict
    pred_map = defaultdict(dict)
    for h, did, pos in zip(hyps, dial_ids, positions):
        pred_map[str(did)][pos - 1] = h   # boundary slot = pos-1

    all_preds, all_labels = [], []
    for d in dialogues:
        key = str(d["dial_id"])
        n   = len(d["utterances"])
        true_bounds = segments_to_boundaries(d["segments"])
        pred_bounds = [pred_map[key].get(i, 0) for i in range(n)]
        all_preds.append(pred_bounds)
        all_labels.append(true_bounds)
    return all_preds, all_labels


# ── table / IO ────────────────────────────────────────────────────────────────

def print_table(all_results):
    col_w  = 10
    header = f"{'Dataset':<14}" + "".join(f"{m:>{col_w}}" for m in METRIC_COLS)
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for name, m in all_results.items():
        print(f"{name:<14}" + "".join(f"{m[c]:>{col_w}.4f}" for c in METRIC_COLS))
    print(sep)


def write_results(all_results, model_name, epochs):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    payload = {"model": model_name, "epochs": epochs, "results": all_results}
    json_path = os.path.join(OUTPUT_ROOT, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    tsv_path = os.path.join(OUTPUT_ROOT, "results.tsv")
    with open(tsv_path, "w") as f:
        f.write("dataset\t" + "\t".join(METRIC_COLS) + "\n")
        for name, m in all_results.items():
            f.write(name + "\t" + "\t".join(f"{m[c]:.4f}" for c in METRIC_COLS) + "\n")
    print(f"\nSaved → {json_path}")
    print(f"Saved → {tsv_path}")


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", default="t5-base")
    p.add_argument("--datasets",      nargs="+", default=["all"])
    p.add_argument("--epochs",        type=int,  default=3)
    p.add_argument("--batch_size",    type=int,  default=64)
    p.add_argument("--lr",            type=float,default=5e-5)
    p.add_argument("--max_input_len", type=int,  default=512)
    p.add_argument("--max_samples",   type=int,  default=0,
                   help="Cap pairs per split per dataset (0 = all)")
    p.add_argument("--few_shot_k",    type=int,  default=0,
                   help="If >0, sample k positive + k negative pairs from train split (test/valid use full data)")
    p.add_argument("--save_ckpt",     action="store_true",
                   help="Save fine-tuned model checkpoint per dataset")
    p.add_argument("--device",        default="auto")
    return p.parse_args()


def main():
    args = parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device : {device}")

    targets = ALL_DATASETS if "all" in args.datasets else args.datasets
    all_results: dict[str, dict] = {}

    for name in targets:
        json_path = os.path.join(DATASET_DIR, f"{name}.json")
        if not os.path.exists(json_path):
            print(f"[SKIP] {name}: not found")
            continue

        print(f"\n{'='*50}")
        print(f"  Dataset : {name}")
        print(f"{'='*50}")

        with open(json_path, encoding="utf-8") as f:
            all_dialogues = json.load(f)
        train_dials = [d for d in all_dialogues if d.get("set") == "train"]
        test_dials  = [d for d in all_dialogues if d.get("set") == "test"]

        # ── fresh model + tokenizer per dataset ──
        print(f"  Loading {args.model_name_or_path} ...")
        tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
        model     = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path).to(device)

        # ── datasets ──
        train_ds = BoundaryDataset(all_dialogues, tokenizer, args.max_input_len,
                                   split="train", few_shot_k=args.few_shot_k)
        test_ds  = BoundaryDataset(all_dialogues, tokenizer, args.max_input_len,
                                   split="test")

        print(f"  Train pairs : {len(train_ds)}  |  Test pairs : {len(test_ds)}")

        # class balance info
        n_pos = sum(train_ds.labels)
        n_neg = len(train_ds.labels) - n_pos
        print(f"  Train balance: pos={n_pos} ({100*n_pos/max(len(train_ds),1):.1f}%)  neg={n_neg}")

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                  num_workers=2, collate_fn=train_collate)
        test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                                  num_workers=2, collate_fn=eval_collate)

        # ── optimizer + scheduler ──
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        total_steps = len(train_loader) * args.epochs
        warmup_steps = max(1, total_steps // 10)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        # ── training loop ──
        t0 = time.time()
        for epoch in range(1, args.epochs + 1):
            avg_loss = train(model, train_loader, optimizer, scheduler, device,
                             epoch, args.epochs)
            print(f"  Epoch {epoch}/{args.epochs}  loss={avg_loss:.4f}")

        # ── test evaluation ──
        print("  Evaluating on test set ...")
        hyps, refs, dids, poss = predict(model, test_loader, tokenizer, device)
        all_preds, all_labels  = reassemble(test_dials, hyps, refs, dids, poss)
        metrics = evaluate_all(all_preds, all_labels)
        all_results[name] = metrics

        elapsed = time.time() - t0
        print(
            f"  PK={metrics['PK']:.4f}  WD={metrics['WD']:.4f}  "
            f"F1={metrics['F1']:.4f}  Score={metrics['Score']:.4f}  "
            f"[{elapsed:.0f}s]"
        )

        # ── optional checkpoint save ──
        if args.save_ckpt:
            ckpt_dir = os.path.join(OUTPUT_ROOT, name, "checkpoint")
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            print(f"  Checkpoint saved → {ckpt_dir}")

        del model, tokenizer
        torch.cuda.empty_cache()

    print_table(all_results)
    write_results(all_results, args.model_name_or_path, args.epochs)


if __name__ == "__main__":
    main()
