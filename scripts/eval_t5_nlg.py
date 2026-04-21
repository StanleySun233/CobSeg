"""
Zero-shot evaluation of t5-base on the five NLG datasets.

For each dataset the script:
  1. Loads the pre-built test TSV from  data/t5_nlg/<dataset>/test.tsv
  2. Runs beam-search generation with the vanilla t5-base checkpoint
     (no fine-tuning, no special tokens needed)
  3. Computes BLEU-1/2/3/4 (sacrebleu corpus), ROUGE-L (rouge-score),
     and METEOR (nltk) against the reference responses
  4. Prints a per-dataset summary table and writes results to
     data/t5_nlg/results.json  (and  results.tsv for copy-paste)

Usage
-----
  # evaluate all five datasets (default)
  python scripts/eval_t5_nlg.py

  # evaluate a subset
  python scripts/eval_t5_nlg.py --datasets vhf tiage

  # use a different checkpoint (e.g. a fine-tuned model)
  python scripts/eval_t5_nlg.py --model_name_or_path t5-large

  # limit rows per dataset for a quick smoke-test
  python scripts/eval_t5_nlg.py --max_samples 200

Options
-------
  --model_name_or_path   HuggingFace model id or local path  [default: t5-base]
  --datasets             one or more dataset names, or "all"  [default: all]
  --batch_size           generation batch size                [default: 64]
  --max_input_len        tokeniser max length for context     [default: 512]
  --max_new_tokens       max tokens to generate per response  [default: 64]
  --num_beams            beam width                           [default: 5]
  --max_samples          cap rows per dataset (0 = unlimited) [default: 0]
  --output_dir           where to write results               [default: data/t5_nlg]
  --device               "cuda" / "cpu" / "auto"             [default: auto]
"""

import argparse
import json
import os
import re
import time

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

import sacrebleu
from rouge_score import rouge_scorer as rouge_scorer_lib
import nltk

nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.translate.meteor_score import meteor_score

# ── constants ────────────────────────────────────────────────────────────────

ALL_DATASETS = ["tiage", "dialseg711", "doc2dial", "vhf", "superseg"]
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(SCRIPT_DIR, "../data/t5_nlg")


# ── dataset ──────────────────────────────────────────────────────────────────

class NLGTestDataset(Dataset):
    """Read a TSV file: context<TAB>reference and tokenise the context."""

    def __init__(self, tsv_path: str, tokenizer, max_input_len: int, max_samples: int = 0):
        self.tokenizer    = tokenizer
        self.max_input_len = max_input_len
        self.contexts: list[str] = []
        self.references: list[str] = []

        with open(tsv_path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                self.contexts.append(parts[0].strip())
                self.references.append(parts[1].strip())
                if max_samples and len(self.contexts) >= max_samples:
                    break

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.contexts[idx],
            max_length=self.max_input_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "reference":      self.references[idx],
        }


def collate_fn(batch):
    return {
        "input_ids":      torch.stack([b["input_ids"]      for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "references":     [b["reference"] for b in batch],
    }


# ── generation ───────────────────────────────────────────────────────────────

def generate_predictions(
    model,
    tokenizer,
    dataset: NLGTestDataset,
    batch_size: int,
    max_new_tokens: int,
    num_beams: int,
    device: torch.device,
) -> tuple[list[str], list[str]]:
    """Return (hypotheses, references) aligned lists."""

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    all_hyps, all_refs = [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  generating", leave=False):
            outs = model.generate(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=3,
                early_stopping=True,
            )
            decoded = tokenizer.batch_decode(outs, skip_special_tokens=True)
            all_hyps.extend(decoded)
            all_refs.extend(batch["references"])

    return all_hyps, all_refs


# ── metrics ──────────────────────────────────────────────────────────────────

def compute_metrics(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Compute BLEU-1/2/3/4, ROUGE-L, METEOR."""

    # BLEU (sacrebleu corpus-level, tokenised internally)
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    bleu_scores = {f"BLEU-{i+1}": bleu.precisions[i] for i in range(4)}

    # ROUGE-L
    scorer = rouge_scorer_lib.RougeScorer(["rougeL"], use_stemmer=False)
    rouge_l_scores = [
        scorer.score(ref, hyp)["rougeL"].fmeasure
        for hyp, ref in zip(hyps, refs)
    ]
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores) * 100

    # METEOR (nltk, token-level)
    meteor_scores = [
        meteor_score([ref.split()], hyp.split())
        for hyp, ref in zip(hyps, refs)
    ]
    meteor = sum(meteor_scores) / len(meteor_scores) * 100

    return {**bleu_scores, "ROUGE-L": rouge_l, "METEOR": meteor}


# ── formatting ───────────────────────────────────────────────────────────────

METRIC_COLS = ["BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "METEOR"]

def print_table(all_results: dict[str, dict]):
    col_w = 9
    header = f"{'Dataset':<14}" + "".join(f"{m:>{col_w}}" for m in METRIC_COLS)
    sep    = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for name, metrics in all_results.items():
        row = f"{name:<14}" + "".join(f"{metrics[m]:>{col_w}.2f}" for m in METRIC_COLS)
        print(row)
    print(sep)


def write_results(all_results: dict, output_dir: str, model_name: str):
    os.makedirs(output_dir, exist_ok=True)

    # JSON
    json_path = os.path.join(output_dir, "results.json")
    payload = {"model": model_name, "results": all_results}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {json_path}")

    # TSV (copy-paste friendly)
    tsv_path = os.path.join(output_dir, "results.tsv")
    with open(tsv_path, "w") as f:
        f.write("dataset\t" + "\t".join(METRIC_COLS) + "\n")
        for name, metrics in all_results.items():
            row = name + "\t" + "\t".join(f"{metrics[m]:.4f}" for m in METRIC_COLS)
            f.write(row + "\n")
    print(f"Results saved → {tsv_path}")


# ── main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Zero-shot T5-base NLG evaluation")
    p.add_argument("--model_name_or_path", default="t5-base")
    p.add_argument("--datasets",    nargs="+", default=["all"])
    p.add_argument("--batch_size",  type=int,  default=64)
    p.add_argument("--max_input_len",  type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--num_beams",   type=int,  default=5)
    p.add_argument("--max_samples", type=int,  default=0,
                   help="Cap rows per dataset; 0 = use all")
    p.add_argument("--output_dir",  default=DEFAULT_DATA)
    p.add_argument("--device",      default="auto",
                   choices=["auto", "cuda", "cpu"])
    return p.parse_args()


def main():
    args = parse_args()

    # device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # datasets
    targets = ALL_DATASETS if "all" in args.datasets else args.datasets

    # model & tokenizer (loaded once, shared across datasets)
    print(f"\nLoading model: {args.model_name_or_path}")
    tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
    model     = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model     = model.to(device)
    print(f"Model loaded ({sum(p.numel() for p in model.parameters())/1e6:.0f}M params)\n")

    all_results = {}

    for name in targets:
        tsv_path = os.path.join(args.output_dir, name, "test.tsv")
        if not os.path.exists(tsv_path):
            print(f"[SKIP] {name}: test.tsv not found at {tsv_path}")
            continue

        print(f"── {name} ──")
        t0 = time.time()

        dataset = NLGTestDataset(
            tsv_path, tokenizer,
            max_input_len=args.max_input_len,
            max_samples=args.max_samples,
        )
        print(f"  {len(dataset)} test samples")

        hyps, refs = generate_predictions(
            model, tokenizer, dataset,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            device=device,
        )

        metrics = compute_metrics(hyps, refs)
        all_results[name] = metrics

        elapsed = time.time() - t0
        metric_str = "  ".join(f"{k}={v:.2f}" for k, v in metrics.items())
        print(f"  {metric_str}  [{elapsed:.0f}s]\n")

    print_table(all_results)
    write_results(all_results, args.output_dir, args.model_name_or_path)


if __name__ == "__main__":
    main()
