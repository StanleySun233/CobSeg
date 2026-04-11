#!/usr/bin/env python3
"""
Analyse Utterance Length and Dialogue Length distributions for all three
datasets (vhf / dialseg711 / doc2dial) and produce:

  1. A console table of every-10th-percentile values + coverage at common thresholds.
  2. A figure with CDF overlays and percentile bar charts.
  3. Per-dataset recommended max_utt_tokens / max_utterances (p95, power-of-2 ceil).

Usage
-----
# Fast: word-split token count
python scripts/analyse_lengths.py

# Accurate: subword token count from a real tokenizer
python scripts/analyse_lengths.py --encoder BAAI/bge-m3
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

from utils.dialogue_dataset import DialogueDataset
from utils.utils import resolve_dataset_path

# ── constants ────────────────────────────────────────────────────────────────

DEFAULT_DATASETS = ["vhf", "dialseg711", "doc2dial"]
COLOURS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]
PCTS       = list(range(0, 101, 10))

UTT_THRESHOLDS  = [32, 48, 64, 96, 128, 192, 256]
DIAL_THRESHOLDS = [16, 32, 48, 64, 96, 128, 192]


# ── helpers ──────────────────────────────────────────────────────────────────

def _ceil_pow2(x: int) -> int:
    """Smallest power of 2 >= x."""
    return 1 << math.ceil(math.log2(max(x, 1)))


def _count_tokens(text: str, tokenizer) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.tokenize(text))


def load_stats(ds_name: str, tokenizer):
    """Return (utt_lengths, dial_lengths) arrays for a dataset (all splits)."""
    ds_path = resolve_dataset_path(ds_name)
    dataset = DialogueDataset(ds_path)

    utt_lengths, dial_lengths = [], []
    for d in tqdm(dataset, desc=f"{ds_name:12s}", leave=False, ncols=70):
        utts = d.utterances
        if not utts:
            continue
        dial_lengths.append(len(utts))
        for u in utts:
            utt_lengths.append(_count_tokens(u, tokenizer))

    return np.array(utt_lengths, dtype=np.float32), np.array(dial_lengths, dtype=np.float32)


# ── print helpers ────────────────────────────────────────────────────────────

def _pct_row(arr: np.ndarray) -> list:
    return [round(float(np.percentile(arr, p))) for p in PCTS]


def print_table(stats: dict, dataset_names: list[str]):
    """Print the percentile table to stdout."""
    col_w = 6
    pct_header = "".join(f"{'p'+str(p):>{col_w}}" for p in PCTS)
    row_fmt = (
        "{:<12s} {:<9s}"
        + (f" {{:>{col_w}d}}" * len(PCTS))
        + "  mean={:6.1f}  std={:6.1f}  n={:,}"
    )

    header = f"{'dataset':<12s} {'metric':<9s}{pct_header}  {'mean':>10}  {'std':>8}  {'n':>8}"
    sep    = "─" * len(header)
    print()
    print(sep)
    print(header)
    print(sep)
    for ds_name in dataset_names:
        for metric, arr in [("utt_tok", stats[ds_name]["utt"]),
                            ("dial_utt", stats[ds_name]["dial"])]:
            row = _pct_row(arr)
            print(row_fmt.format(
                ds_name, metric, *row,
                float(arr.mean()), float(arr.std()), len(arr),
            ))
    print(sep)


def print_coverage(stats: dict, dataset_names: list[str]):
    """For each dataset, show what % of samples fall within each threshold."""
    def cov(arr, thresh):
        return 100.0 * (arr <= thresh).sum() / len(arr)

    print()
    print("=== Coverage at common thresholds ===")
    print()

    for metric, thresholds in [("utt_tok", UTT_THRESHOLDS), ("dial_utt", DIAL_THRESHOLDS)]:
        th_header = "".join(f"{'@'+str(t):>7}" for t in thresholds)
        print(f"  {metric}   {th_header}")
        print("  " + "─" * (14 + 7 * len(thresholds)))
        for ds_name in dataset_names:
            arr = stats[ds_name]["utt" if metric == "utt_tok" else "dial"]
            vals = "".join(f"{cov(arr, t):>6.1f}%" for t in thresholds)
            print(f"  {ds_name:<12s}{vals}")
        print()


def print_recommendations(stats: dict, dataset_names: list[str]):
    """Recommend max_utt_tokens and max_utterances per dataset."""
    print("=== Recommended per-dataset limits (p95, ceil to power-of-2) ===")
    print()
    for ds_name in dataset_names:
        u95 = int(np.percentile(stats[ds_name]["utt"],  95))
        d95 = int(np.percentile(stats[ds_name]["dial"], 95))
        rec_tok = _ceil_pow2(u95)
        rec_utt = _ceil_pow2(d95)
        u99 = int(np.percentile(stats[ds_name]["utt"],  99))
        d99 = int(np.percentile(stats[ds_name]["dial"], 99))
        print(f"  {ds_name:<12s}  "
              f"max_utt_tokens={rec_tok:>4d}  (p95={u95:>4d}, p99={u99:>4d})   "
              f"max_utterances={rec_utt:>4d}  (p95={d95:>4d}, p99={d99:>4d})")
    print()


# ── figure ───────────────────────────────────────────────────────────────────

def _cdf(arr: np.ndarray, xmax: float):
    x = np.sort(arr)
    x = x[x <= xmax]
    y = np.arange(1, len(x) + 1) / len(arr) * 100.0
    return x, y


def make_figure(stats: dict, out_path: Path, token_label: str, dataset_names: list[str]):
    """Four-panel figure: CDF × 2 (top) + percentile bar chart × 2 (bottom)."""
    fig, axes = plt.subplots(
        2, 2, figsize=(14, 9),
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )
    (ax_utt_cdf, ax_dial_cdf), (ax_utt_bar, ax_dial_bar) = axes

    bar_x = np.arange(len(PCTS))
    n_ds = len(dataset_names)
    bar_w = min(0.28, 0.8 / max(n_ds, 1))
    offsets = [(i - (n_ds - 1) / 2.0) * bar_w for i in range(n_ds)]
    colours = [COLOURS[i % len(COLOURS)] for i in range(n_ds)]

    # ── CDFs ─────────────────────────────────────────────────────────────────
    for ds_name, colour in zip(dataset_names, colours):
        utt_arr  = stats[ds_name]["utt"]
        dial_arr = stats[ds_name]["dial"]

        # percentile cut-off for readable axis
        xmax_u = float(np.percentile(utt_arr,  99.5))
        xmax_d = float(np.percentile(dial_arr, 99.5))

        xu, yu = _cdf(utt_arr,  xmax_u)
        xd, yd = _cdf(dial_arr, xmax_d)

        ax_utt_cdf.plot(xu,  yu,  color=colour, lw=1.8, label=ds_name)
        ax_dial_cdf.plot(xd, yd, color=colour, lw=1.8, label=ds_name)

    # mark common thresholds on CDF plots
    for t in UTT_THRESHOLDS:
        ax_utt_cdf.axvline(t, color="grey", lw=0.7, linestyle=":")
        ax_utt_cdf.text(t + 0.5, 4, str(t), fontsize=6.5, color="grey", rotation=90, va="bottom")
    for t in DIAL_THRESHOLDS:
        ax_dial_cdf.axvline(t, color="grey", lw=0.7, linestyle=":")
        ax_dial_cdf.text(t + 0.3, 4, str(t), fontsize=6.5, color="grey", rotation=90, va="bottom")

    # horizontal reference lines at 90%, 95%, 99%
    for ax in (ax_utt_cdf, ax_dial_cdf):
        for pct, ls in [(90, "--"), (95, "-."), (99, ":")]:
            ax.axhline(pct, color="#555", lw=0.8, linestyle=ls, alpha=0.6)
            ax.text(1, pct + 0.8, f"{pct}%", fontsize=6.5, color="#555")

    ax_utt_cdf.set_xlabel(f"Utterance length  ({token_label})", fontsize=9)
    ax_utt_cdf.set_ylabel("Cumulative % of utterances", fontsize=9)
    ax_utt_cdf.set_title("Utterance Length — CDF", fontsize=10, fontweight="bold")
    ax_utt_cdf.set_ylim(0, 102)
    ax_utt_cdf.legend(fontsize=8)
    ax_utt_cdf.yaxis.set_major_formatter(ticker.PercentFormatter())

    ax_dial_cdf.set_xlabel("Dialogue length  (# utterances)", fontsize=9)
    ax_dial_cdf.set_ylabel("Cumulative % of dialogues", fontsize=9)
    ax_dial_cdf.set_title("Dialogue Length — CDF", fontsize=10, fontweight="bold")
    ax_dial_cdf.set_ylim(0, 102)
    ax_dial_cdf.legend(fontsize=8)
    ax_dial_cdf.yaxis.set_major_formatter(ticker.PercentFormatter())

    # ── percentile bar charts ─────────────────────────────────────────────────
    for ds_name, colour, offset in zip(dataset_names, colours, offsets):
        utt_pct  = _pct_row(stats[ds_name]["utt"])
        dial_pct = _pct_row(stats[ds_name]["dial"])

        ax_utt_bar.bar(bar_x + offset, utt_pct,  width=bar_w * 0.9,
                       color=colour, alpha=0.82, label=ds_name)
        ax_dial_bar.bar(bar_x + offset, dial_pct, width=bar_w * 0.9,
                        color=colour, alpha=0.82, label=ds_name)

        # annotate the last bar (p100 = max)
        ax_utt_bar.text(bar_x[-1] + offset, utt_pct[-1] + 0.5,
                        str(utt_pct[-1]), ha="center", va="bottom", fontsize=6)
        ax_dial_bar.text(bar_x[-1] + offset, dial_pct[-1] + 0.3,
                         str(dial_pct[-1]), ha="center", va="bottom", fontsize=6)

    for ax in (ax_utt_bar, ax_dial_bar):
        ax.set_xticks(bar_x)
        ax.set_xticklabels([f"p{p}" for p in PCTS], fontsize=7.5)
        ax.set_ylabel("Value", fontsize=9)
        ax.set_xlabel("Percentile", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)

    ax_utt_bar.set_title("Utterance Length — Every-10th Percentile", fontsize=10, fontweight="bold")
    ax_dial_bar.set_title("Dialogue Length — Every-10th Percentile", fontsize=10, fontweight="bold")

    fig.suptitle("Dataset Length Statistics  (all splits combined)", fontsize=12, y=1.01)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved → {out_path}")


# ── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Dataset length statistics")
    parser.add_argument(
        "--encoder", default=None,
        help="HF model name for subword tokenisation (e.g. BAAI/bge-m3). "
             "If omitted, word-split count is used (faster).",
    )
    parser.add_argument(
        "--out_dir", default=None,
        help="Output directory (default: scripts/output/)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Dataset names (resolve_dataset_path). Default: vhf dialseg711 doc2dial",
    )
    args = parser.parse_args()
    dataset_names = args.datasets if args.datasets else list(DEFAULT_DATASETS)

    # ── optional tokenizer ────────────────────────────────────────────────────
    tokenizer = None
    token_label = "words"
    if args.encoder:
        from transformers import AutoTokenizer
        print(f"Loading tokenizer: {args.encoder}")
        tokenizer = AutoTokenizer.from_pretrained(args.encoder, trust_remote_code=True, local_files_only=True)
        token_label = "subword tokens"

    # ── collect stats ─────────────────────────────────────────────────────────
    stats = {}
    for ds_name in dataset_names:
        utt_arr, dial_arr = load_stats(ds_name, tokenizer)
        stats[ds_name] = {"utt": utt_arr, "dial": dial_arr}
        print(f"  {ds_name:<12s}  "
              f"utterances={len(utt_arr):,}  dialogues={len(dial_arr):,}")

    # ── console output ────────────────────────────────────────────────────────
    print_table(stats, dataset_names)
    print_coverage(stats, dataset_names)
    print_recommendations(stats, dataset_names)

    # ── figure ────────────────────────────────────────────────────────────────
    out_dir  = Path(args.out_dir) if args.out_dir else ROOT / "scripts" / "output"
    suffix   = "_subword" if tokenizer else "_word"
    ds_tag = "_".join(dataset_names) if len(dataset_names) <= 3 else f"{len(dataset_names)}ds"
    out_path = out_dir / f"dataset_length_stats{suffix}_{ds_tag}.png"
    make_figure(stats, out_path, token_label, dataset_names)


if __name__ == "__main__":
    main()
