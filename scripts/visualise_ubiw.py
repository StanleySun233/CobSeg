#!/usr/bin/env python3
"""
Visualise per-utterance UBIW weights + per-token boundary attribution for a
single dialogue.

Each utterance is rendered as a row of words.  Word backgrounds are shaded by
their gradient attribution to the boundary logit (darker = stronger
contribution to a cut point).  The right sidebar shows per-utterance UBIW
weights.

Usage
-----
python scripts/visualise_ubiw.py \
    --dataset vhf \
    --dialogue_id 0 \
    --exp_name topic_kw_v1 \
    --encoder BAAI/bge-m3
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib import cm
from transformers import AutoModel, AutoTokenizer

from utils.dialogue_dataset import DialogueDataset
from utils.utils import resolve_dataset_path
from utils.dts_data import (
    encode_utterances_hf,
    _build_kw_channel_scores,
    topic_channel_sets_from_info,
    MAX_UTT_TOKENS,
    MAX_UTTERANCES,
)
from utils.dts_utils import segments_to_boundaries
from model.dud import DUD

CS_ENCODER_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"


# ── helpers ────────────────────────────────────────────────────────────────────

def _find_dialogue(dialogues: list, dialogue_id: str):
    """Return (dialogue, local_index). Accepts int index or dial_id string."""
    try:
        idx = int(dialogue_id)
        if 0 <= idx < len(dialogues):
            return dialogues[idx], idx
        raise IndexError(f"Index {idx} out of range for split (0–{len(dialogues) - 1})")
    except ValueError:
        pass
    for i, d in enumerate(dialogues):
        if str(d.dial_id) == str(dialogue_id):
            return d, i
    raise ValueError(
        f"Dialogue '{dialogue_id}' not found in selected split. "
        f"Use an integer index or a valid dial_id string."
    )


def _auto_ckpt(dataset: str, exp_name: str) -> Path:
    ds_path = resolve_dataset_path(dataset)
    stem = Path(ds_path).stem
    return ROOT / "checkpoints" / stem / exp_name / "best.pt"


def _load_topic_channels(topic_json_path: str, dataset: str) -> dict[str, set[str]]:
    p = Path(topic_json_path)
    if not p.exists():
        return topic_channel_sets_from_info({})
    with open(p) as f:
        topic_data = json.load(f)
    ds_path = resolve_dataset_path(dataset)
    stem = Path(ds_path).stem
    for key in (stem, dataset, dataset.lower()):
        if key in topic_data:
            return topic_channel_sets_from_info(topic_data[key])
    return topic_channel_sets_from_info({})


# ── token → word attribution ───────────────────────────────────────────────────

def _word_ids_for_utt(tokenizer, text: str, max_utt_tokens: int) -> list | None:
    """
    Return a list of word-ids (one per subword token) using the fast-tokenizer
    API.  Returns None if the tokenizer does not support it.
    """
    try:
        enc = tokenizer(text, truncation=True, max_length=max_utt_tokens)
        wids = enc.word_ids()   # available on PreTrainedTokenizerFast
        return wids             # list[int | None], None = special token
    except Exception:
        return None


def _compute_word_attributions(
    model: torch.nn.Module,
    tokenizer,
    utts: list[str],
    x_s: torch.Tensor,
    x_w: torch.Tensor,
    tok_mask: torch.Tensor,
    x_t: torch.Tensor,
    lengths: torch.Tensor,
    kw_t: torch.Tensor,
    n: int,
    max_utt_tokens: int,
    attr_target: str = "cut",
) -> tuple[list[list[str]], list[np.ndarray]]:
    """
    Gradient attribution of each subword token w.r.t. the sum of boundary
    logits (class-1 emissions) for the dialogue, then aggregated to word level.

    The proxy is  attr_j = ||∂L/∂x_j  ⊙  x_j||₁   (integrated-gradient-like).

    Returns
    -------
    words_per_utt      : list[list[str]] – whitespace-split words per utterance
    word_attrs_per_utt : list[np.ndarray] – attribution in [0,1] per word
    """
    was_training = model.training
    model.train()

    # Gradient w.r.t. token stream input
    x_w_g = x_w.clone().detach().requires_grad_(True)

    with torch.enable_grad():
        model.zero_grad(set_to_none=True)
        if hasattr(model, "forward_heads"):
            cut_emissions, end_emissions, start_emissions, _, _ = model.forward_heads(
                x_s, x_w_g, tok_mask, x_t, lengths, kw_scores=kw_t
            )
            if attr_target == "end":
                target_logits = end_emissions
            elif attr_target == "start":
                target_logits = start_emissions
            else:
                target_logits = cut_emissions
        else:
            target_logits, _ = model(x_s, x_w_g, tok_mask, x_t, lengths, kw_scores=kw_t)
        target = target_logits[0, :n, 1].sum()
        target.backward()

    if not was_training:
        model.eval()

    # grad shape: (1, T_max, L, d)  →  take valid utterances
    grad = x_w_g.grad[0, :n, :, :].detach().cpu()    # (n, L, d)
    inp  = x_w[0, :n, :, :].detach().cpu()            # (n, L, d)
    # |grad ⊙ inp| summed over embedding dimension → scalar per token position
    tok_attr = (grad * inp).abs().sum(dim=-1).numpy()  # (n, L)

    words_per_utt: list[list[str]] = []
    word_attrs_per_utt: list[np.ndarray] = []

    for i, utt in enumerate(utts):
        words = utt.split()
        if not words:
            words_per_utt.append(["<empty>"])
            word_attrs_per_utt.append(np.zeros(1))
            continue

        word_ids = _word_ids_for_utt(tokenizer, utt, max_utt_tokens)

        if word_ids is not None:
            # Aggregate subword attributions to word level (mean pooling)
            n_words  = len(words)
            w_attr   = np.zeros(n_words)
            w_count  = np.zeros(n_words)
            for j, wid in enumerate(word_ids):
                if wid is not None and wid < n_words:
                    w_attr[wid]  += tok_attr[i, j]
                    w_count[wid] += 1
            valid = w_count > 0
            w_attr[valid] /= w_count[valid]
        else:
            # Fallback: evenly distribute token attribution across words
            n_toks = int(tok_mask[0, i, :].sum().item())
            n_words = len(words)
            w_attr = np.zeros(n_words)
            if n_words > 0 and n_toks > 0:
                toks_per_word = n_toks / n_words
                for wi in range(n_words):
                    t_start = round(wi * toks_per_word)
                    t_end   = round((wi + 1) * toks_per_word)
                    t_end   = min(t_end, n_toks)
                    if t_start < t_end:
                        w_attr[wi] = tok_attr[i, t_start:t_end].mean()

        # Normalise per utterance to [0, 1]
        vmax = w_attr.max()
        if vmax > 1e-9:
            w_attr = w_attr / vmax

        words_per_utt.append(words)
        word_attrs_per_utt.append(w_attr)

    return words_per_utt, word_attrs_per_utt


def _distance_decay(boundaries: list[int], n: int, tau: float) -> np.ndarray:
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    b = np.where(np.array(boundaries[:n], dtype=np.int64) == 1)[0]
    if b.size == 0:
        return np.ones(n, dtype=np.float32)
    pos = np.arange(n, dtype=np.float32)[:, None]
    dist = np.abs(pos - b[None, :]).min(axis=1)
    return np.exp(-dist / max(float(tau), 1e-6)).astype(np.float32)


# ── rendering ──────────────────────────────────────────────────────────────────

_CMAP_TOK  = cm.get_cmap("YlOrRd")   # word attribution heat (yellow → red)
_CMAP_UBIW = cm.get_cmap("Blues")    # UBIW sidebar (light → dark blue)
_NORM_01   = Normalize(vmin=0.0, vmax=1.0)

# Approximate character width in axes-normalised units (monospace, fontsize≈7.5)
_CHAR_W   = 0.0092
_SPACE_W  = 0.007
_MAX_WORDS = 22   # clip very long utterances so they fit the row


def _draw_word_row(
    ax,
    row_y: float,
    words: list[str],
    attrs: np.ndarray,
    fontsize: float = 7.5,
) -> None:
    """
    Render one utterance as a horizontal sequence of colour-coded word boxes.
    Words with high attribution get a dark-red background; low-attribution
    words are pale yellow.  Overflowing words are clipped (clip_on=True).
    """
    x = 0.015
    for word, a in zip(words, attrs):
        if x > 0.965:
            # Indicate truncation
            ax.text(0.970, row_y, "…", fontsize=fontsize, va="center",
                    ha="left", color="#888888", clip_on=True)
            break
        rgba = _CMAP_TOK(_NORM_01(float(a)))
        # Contrast-aware text colour (dark bg → white text)
        r, g, b, _ = rgba
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        txt_color = "black" if lum > 0.42 else "white"
        ax.text(
            x, row_y, word,
            fontsize=fontsize,
            fontfamily="monospace",
            color=txt_color,
            va="center", ha="left",
            bbox=dict(
                facecolor=rgba,
                edgecolor="none",
                boxstyle="round,pad=0.15",
                alpha=0.93,
            ),
            clip_on=True,
        )
        x += len(word) * _CHAR_W + _SPACE_W


# ── plot ───────────────────────────────────────────────────────────────────────

def _plot(
    utts: list[str],
    ubiw_end_disp_01: np.ndarray,
    ubiw_start_disp_01: np.ndarray,
    ubiw_disp_label: str,
    words_per_utt: list[list[str]],
    word_attrs_per_utt: list[np.ndarray],
    gt_boundaries: list[int],
    pred_boundaries: list[int],
    dial_id: str,
    dataset: str,
    out_dir: Path,
) -> Path:
    n = len(utts)
    out_dir.mkdir(parents=True, exist_ok=True)

    row_h = max(0.32, min(0.55, 11.0 / n))
    fig_h = max(6.0, n * row_h + 2.5)

    fig, (ax_tok, ax_ubiw_end, ax_ubiw_start) = plt.subplots(
        1, 3,
        figsize=(20, fig_h),
        gridspec_kw={"width_ratios": [6, 1, 1]},
        constrained_layout=True,
    )

    # ── alternating segment background ─────────────────────────────────────────
    seg_id = 0
    for i in range(n):
        shade = 0.93 if seg_id % 2 == 0 else 1.00
        ax_tok.barh(i, 1.0, left=0, color=(shade, shade, shade),
                    height=1.0, align="center", zorder=0)
        if i < len(gt_boundaries) and gt_boundaries[i] == 1:
            seg_id += 1

    # ── word-level attribution rows ─────────────────────────────────────────────
    for i in range(n):
        words = words_per_utt[i][:_MAX_WORDS]
        attrs = word_attrs_per_utt[i][:_MAX_WORDS]
        _draw_word_row(ax_tok, i, words, attrs)

    # ── utterance index labels (left edge) ─────────────────────────────────────
    for i in range(n):
        suffix = " GT▶" if (i < len(gt_boundaries) and gt_boundaries[i] == 1) else ""
        ax_tok.text(
            -0.002, i, f"[{i:02d}]{suffix}",
            fontsize=6.8, va="center", ha="right",
            fontfamily="monospace", color="#444444",
            transform=ax_tok.get_yaxis_transform(),
        )

    # ── boundary lines ──────────────────────────────────────────────────────────
    gt_drawn = pred_drawn = False
    for i, b in enumerate(gt_boundaries):
        if b == 1:
            ax_tok.axhline(
                y=i + 0.5, color="#1565C0", lw=2.4, ls="--", zorder=6,
                label="GT boundary" if not gt_drawn else "_",
            )
            gt_drawn = True
    for i, b in enumerate(pred_boundaries):
        if b == 1:
            ax_tok.axhline(
                y=i + 0.5, color="#C62828", lw=1.8, ls="-", zorder=7, alpha=0.88,
                label="Pred boundary" if not pred_drawn else "_",
            )
            pred_drawn = True

    # ── main axes formatting ────────────────────────────────────────────────────
    ax_tok.set_xlim(0, 1)
    ax_tok.set_ylim(n - 0.5, -0.5)
    ax_tok.set_yticks([])
    ax_tok.set_xticks([])
    ax_tok.set_xlabel(
        "← word colour: boundary attribution  (YlOrRd: low → high) →",
        fontsize=8.5,
    )
    ax_tok.set_title(
        f"Token Attribution Heatmap   dataset={dataset}   dial_id={dial_id}",
        fontsize=11, fontweight="bold", pad=8,
    )

    sm_tok = plt.cm.ScalarMappable(cmap=_CMAP_TOK, norm=_NORM_01)
    sm_tok.set_array([])
    cbar = fig.colorbar(sm_tok, ax=ax_tok, shrink=0.45, pad=0.012, aspect=22)
    cbar.set_label("Token attribution (norm. per utterance)", fontsize=7.5)
    cbar.ax.tick_params(labelsize=7)

    handles = []
    if gt_drawn:
        handles.append(mpatches.Patch(facecolor="#1565C0", label="GT boundary  (--)"))
    if pred_drawn:
        handles.append(mpatches.Patch(facecolor="#C62828", label="Pred boundary (—)"))
    if handles:
        ax_tok.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.85)

    # ── UBIW sidebar ────────────────────────────────────────────────────────────
    ubiw_norm = Normalize(vmin=0.0, vmax=1.0)
    ubiw_cmap = _CMAP_UBIW

    for i in range(n):
        v = float(ubiw_end_disp_01[i])
        color = ubiw_cmap(ubiw_norm(v))
        ax_ubiw_end.barh(i, v, color=color, height=0.75, align="center")
    for i in range(n):
        v = float(ubiw_start_disp_01[i])
        color = ubiw_cmap(ubiw_norm(v))
        ax_ubiw_start.barh(i, v, color=color, height=0.75, align="center")

    for i, b in enumerate(gt_boundaries):
        if b == 1:
            ax_ubiw_end.axhline(y=i + 0.5, color="#1565C0", lw=2.0, ls="--", zorder=5)
            ax_ubiw_start.axhline(y=i + 0.5, color="#1565C0", lw=2.0, ls="--", zorder=5)
    for i, b in enumerate(pred_boundaries):
        if b == 1:
            ax_ubiw_end.axhline(y=i + 0.5, color="#C62828", lw=1.6, ls="-", zorder=6, alpha=0.88)
            ax_ubiw_start.axhline(y=i + 0.5, color="#C62828", lw=1.6, ls="-", zorder=6, alpha=0.88)

    ax_ubiw_end.set_ylim(n - 0.5, -0.5)
    ax_ubiw_end.set_yticks([])
    ax_ubiw_end.set_xlim(0, 1.05)
    ax_ubiw_end.set_xlabel(f"UBIW end\n{ubiw_disp_label} (0-1)", fontsize=8)
    ax_ubiw_end.set_title("UBIW End", fontsize=9)
    ax_ubiw_end.tick_params(axis="x", labelsize=7)
    ax_ubiw_start.set_ylim(n - 0.5, -0.5)
    ax_ubiw_start.set_yticks([])
    ax_ubiw_start.set_xlim(0, 1.05)
    ax_ubiw_start.set_xlabel(f"UBIW start\n{ubiw_disp_label} (0-1)", fontsize=8)
    ax_ubiw_start.set_title("UBIW Start", fontsize=9)
    ax_ubiw_start.tick_params(axis="x", labelsize=7)
    cbar_label = f"UBIW {ubiw_disp_label} (min-max 0-1)"

    sm_ubiw = plt.cm.ScalarMappable(cmap=ubiw_cmap, norm=ubiw_norm)
    sm_ubiw.set_array([])
    cbar2 = fig.colorbar(sm_ubiw, ax=[ax_ubiw_end, ax_ubiw_start], shrink=0.45, pad=0.04, aspect=22)
    cbar2.set_label(cbar_label, fontsize=7.5)
    cbar2.ax.tick_params(labelsize=7)

    # ── save ────────────────────────────────────────────────────────────────────
    safe_id = str(dial_id).replace("/", "_").replace(" ", "_")
    fname = out_dir / f"ubiw_token_{dataset}_{safe_id}.png"
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {fname}")
    return fname


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visualise UBIW + token attribution heatmap for one dialogue"
    )
    parser.add_argument("--dataset", default="vhf",
                        help="vhf | dialseg711 | doc2dial  or direct path to .json")
    parser.add_argument("--dialogue_id", default="0",
                        help="Integer index within split, or dial_id string")
    parser.add_argument("--split", default="test",
                        help="Which split to search: train | val | test  (default: test)")
    parser.add_argument("--exp_name", default="kw_rank_v1",
                        help="Experiment name used when training (for checkpoint lookup)")
    parser.add_argument("--encoder", default="BAAI/bge-m3")
    parser.add_argument("--ckpt", default=None,
                        help="Explicit path to best.pt; auto-derived from exp_name if omitted")
    parser.add_argument("--max_utt_tokens", type=int, default=MAX_UTT_TOKENS)
    parser.add_argument("--max_utterances", type=int, default=MAX_UTTERANCES)
    parser.add_argument("--topic_json_path", default="./data/topic/topic_keywords.json")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: scripts/output/)")
    parser.add_argument("--attr_target", choices=("cut", "end", "start"), default="cut")
    parser.add_argument("--ubiw_view", choices=("w", "delta_gain", "zscore"), default="delta_gain")
    parser.add_argument("--token_decay", choices=("none", "gt", "pred"), default="gt")
    parser.add_argument("--token_decay_tau", type=float, default=2.0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── dataset ────────────────────────────────────────────────────────────────
    ds_path = resolve_dataset_path(args.dataset)
    print(f"Loading dataset: {ds_path}")
    full_dataset = DialogueDataset(ds_path)

    split_aliases = {
        "val":   ("valid", "val", "dev"),
        "test":  ("test",),
        "train": ("train",),
    }
    valid_splits = split_aliases.get(args.split, (args.split,))
    dialogues = [d for d in full_dataset if d.set in valid_splits]
    if not dialogues:
        print(f"[warn] No dialogues found for split={args.split!r}; searching all splits.")
        dialogues = list(full_dataset)

    dialogue, local_idx = _find_dialogue(dialogues, args.dialogue_id)
    utts = dialogue.utterances[: args.max_utterances]
    n = len(utts)
    gt_boundaries = segments_to_boundaries(dialogue.segments)[:n]

    print(
        f"Dialogue  dial_id={dialogue.dial_id!r}  split={dialogue.set!r}  "
        f"local_idx={local_idx}  n_utts={n}  segments={dialogue.segments}"
    )

    ckpt_path = Path(args.ckpt) if args.ckpt else _auto_ckpt(args.dataset, args.exp_name)
    if not ckpt_path.exists():
        print(f"[error] Checkpoint not found: {ckpt_path}")
        print("        Train the model first, or pass --ckpt <path>.")
        sys.exit(1)
    print(f"Loading checkpoint: {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "pos_emb" in state_dict:
        ckpt_max_utt_tokens = int(state_dict["pos_emb"].shape[1])
        if ckpt_max_utt_tokens != args.max_utt_tokens:
            print(
                f"[info] Override --max_utt_tokens from {args.max_utt_tokens} "
                f"to checkpoint value {ckpt_max_utt_tokens}"
            )
            args.max_utt_tokens = ckpt_max_utt_tokens

    # ── keyword scores ─────────────────────────────────────────────────────────
    kw_ch = _load_topic_channels(args.topic_json_path, args.dataset)
    kw_scores = _build_kw_channel_scores(utts, kw_ch)

    # ── encode ─────────────────────────────────────────────────────────────────
    print(f"Loading encoder: {args.encoder}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder, trust_remote_code=True, local_files_only=True)
    enc_model = AutoModel.from_pretrained(args.encoder, trust_remote_code=True, local_files_only=True).to(device).eval()
    print(f"Loading CS encoder: {CS_ENCODER_NAME}")
    cs_tokenizer = AutoTokenizer.from_pretrained(CS_ENCODER_NAME, trust_remote_code=True, local_files_only=True)
    cs_enc_model = AutoModel.from_pretrained(CS_ENCODER_NAME, trust_remote_code=True, local_files_only=True).to(device).eval()

    sent_np, tok_np, mask_np = encode_utterances_hf(
        enc_model, tokenizer, utts, device,
        batch_size=32,
        max_utt_tokens=args.max_utt_tokens,
        show_progress=True,
    )
    cs_sent_np, _, _ = encode_utterances_hf(
        cs_enc_model, cs_tokenizer, utts, device,
        batch_size=32,
        max_utt_tokens=args.max_utt_tokens,
        show_progress=True,
    )
    input_dim = sent_np.shape[-1]

    # batch-of-1 tensors
    x_s   = torch.tensor(sent_np,   dtype=torch.float32).unsqueeze(0).to(device)
    x_w   = torch.tensor(tok_np,    dtype=torch.float32).unsqueeze(0).to(device)
    tok_m = torch.tensor(mask_np,   dtype=torch.float32).unsqueeze(0).to(device)
    x_t = torch.tensor(cs_sent_np, dtype=torch.float32).unsqueeze(0).to(device)
    kw_t = kw_scores.unsqueeze(0).to(device)
    lengths = torch.tensor([n], dtype=torch.long)

    # ── load model ─────────────────────────────────────────────────────────────
    model = DUD(
        input_dim=input_dim,
        max_utt_tokens=args.max_utt_tokens,
        use_ubiw=True,
        topic_json_path=args.topic_json_path,
    ).to(device)
    load_res = model.load_state_dict(state_dict, strict=False)
    if hasattr(model, "sync_topic_branch_from_sentence"):
        if any(
            k.startswith("lstm_t")
            or k.startswith("res_t")
            or k.startswith("head_t")
            for k in load_res.missing_keys
        ):
            model.sync_topic_branch_from_sentence()
    if hasattr(model, "sync_start_heads_from_end"):
        if any(k.startswith("head_s_start") or k.startswith("head_w_start") for k in load_res.missing_keys):
            model.sync_start_heads_from_end()
    model.eval()

    ubiw_strength_val = torch.sigmoid(model.ubiw_strength).item()
    print(f"Learned ubiw_strength (sigmoid): {ubiw_strength_val:.4f}")

    # ── inference (UBIW weights + boundary predictions) ───────────────────────
    with torch.no_grad():
        emissions, mask = model(x_s, x_w, tok_m, x_t, lengths, kw_scores=kw_t)
        if hasattr(model, "get_ubiw_weights_dual"):
            ubiw_end_w, ubiw_start_w = model.get_ubiw_weights_dual(x_s, lengths)
        else:
            ubiw_w = model.get_ubiw_weights(x_s, lengths)
            ubiw_end_w, ubiw_start_w = ubiw_w, ubiw_w

    if model.crf is not None:
        pred_seq = model.crf.decode(emissions, mask=mask)[0]
    else:
        pred_seq = emissions[0].argmax(dim=-1).tolist()

    pred_boundaries = [int(p) for p in pred_seq[:n]]
    ubiw_end = ubiw_end_w[0, :n].cpu().numpy()
    ubiw_start = ubiw_start_w[0, :n].cpu().numpy()
    ubiw = 0.5 * (ubiw_end + ubiw_start)

    def _to_view(raw: np.ndarray, view: str):
        v_mean = float(raw.mean())
        v_std = float(raw.std())
        v_delta_gain = ubiw_strength_val * (raw - v_mean)
        if v_std > 1e-9:
            v_z = (raw - v_mean) / v_std
        else:
            v_z = np.zeros_like(raw)
        if view == "w":
            disp = raw
        elif view == "zscore":
            disp = v_z
        else:
            disp = v_delta_gain
        dmin = float(disp.min())
        dmax = float(disp.max())
        if dmax - dmin > 1e-9:
            disp_01 = (disp - dmin) / (dmax - dmin)
        else:
            disp_01 = np.zeros_like(disp)
        return disp, disp_01, v_delta_gain

    ubiw_end_disp, ubiw_end_disp_01, ubiw_end_delta_gain = _to_view(ubiw_end, args.ubiw_view)
    ubiw_start_disp, ubiw_start_disp_01, ubiw_start_delta_gain = _to_view(ubiw_start, args.ubiw_view)
    if args.ubiw_view == "w":
        ubiw_disp_label = "w"
    elif args.ubiw_view == "zscore":
        ubiw_disp_label = "z-score"
    else:
        ubiw_disp_label = "delta_gain"

    n_gt   = sum(gt_boundaries)
    n_pred = sum(pred_boundaries)
    print(f"GT boundaries: {n_gt}  |  Pred boundaries: {n_pred}")
    print(
        f"UBIW weights  min={ubiw.min():.3f}  max={ubiw.max():.3f}  "
        f"mean={ubiw.mean():.3f}  std={ubiw.std():.3f}"
    )
    print(
        f"UBIW_end delta_gain  min={ubiw_end_delta_gain.min():.6f}  max={ubiw_end_delta_gain.max():.6f}  "
        f"mean={ubiw_end_delta_gain.mean():.6f}  std={ubiw_end_delta_gain.std():.6f}"
    )
    print(
        f"UBIW_start delta_gain  min={ubiw_start_delta_gain.min():.6f}  max={ubiw_start_delta_gain.max():.6f}  "
        f"mean={ubiw_start_delta_gain.mean():.6f}  std={ubiw_start_delta_gain.std():.6f}"
    )
    print(
        f"UBIW_end {ubiw_disp_label} -> [0,1]  min={ubiw_end_disp_01.min():.6f}  "
        f"max={ubiw_end_disp_01.max():.6f}  mean={ubiw_end_disp_01.mean():.6f}  std={ubiw_end_disp_01.std():.6f}"
    )
    print(
        f"UBIW_start {ubiw_disp_label} -> [0,1]  min={ubiw_start_disp_01.min():.6f}  "
        f"max={ubiw_start_disp_01.max():.6f}  mean={ubiw_start_disp_01.mean():.6f}  std={ubiw_start_disp_01.std():.6f}"
    )

    top5_idx = np.argsort(0.5 * (ubiw_end_disp + ubiw_start_disp))[::-1][:5]
    print(f"Top-5 utterances by avg({ubiw_disp_label}_end, {ubiw_disp_label}_start):")
    for rank, idx in enumerate(top5_idx, 1):
        print(
            f"  #{rank}  [utt {idx:02d}]  w_end={ubiw_end[idx]:.6f}  w_start={ubiw_start[idx]:.6f}  "
            f"{ubiw_disp_label}_end={ubiw_end_disp[idx]:.6f}  {ubiw_disp_label}_start={ubiw_start_disp[idx]:.6f}  "
            f"gt={gt_boundaries[idx]}  pred={pred_boundaries[idx]}  "
            f"text={utts[idx][:70]!r}"
        )

    # ── token attribution ──────────────────────────────────────────────────────
    print("Computing token-level gradient attribution …")
    words_per_utt, word_attrs_per_utt = _compute_word_attributions(
        model, tokenizer, utts,
        x_s, x_w, tok_m, x_t, lengths, kw_t,
        n, args.max_utt_tokens,
        attr_target=args.attr_target,
    )
    if args.token_decay != "none":
        if args.token_decay == "pred":
            decay = _distance_decay(pred_boundaries, n, args.token_decay_tau)
        else:
            decay = _distance_decay(gt_boundaries, n, args.token_decay_tau)
        for i in range(n):
            word_attrs_per_utt[i] = np.clip(word_attrs_per_utt[i] * float(decay[i]), 0.0, 1.0)
        print(
            f"Token decay mode={args.token_decay} tau={args.token_decay_tau:.3f}  "
            f"min={decay.min():.3f} max={decay.max():.3f} mean={decay.mean():.3f}"
        )

    # ── plot ───────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "scripts" / "output"
    _plot(
        utts=utts,
        ubiw_end_disp_01=ubiw_end_disp_01,
        ubiw_start_disp_01=ubiw_start_disp_01,
        ubiw_disp_label=ubiw_disp_label,
        words_per_utt=words_per_utt,
        word_attrs_per_utt=word_attrs_per_utt,
        gt_boundaries=gt_boundaries,
        pred_boundaries=pred_boundaries,
        dial_id=dialogue.dial_id,
        dataset=args.dataset,
        out_dir=out_dir,
    )


if __name__ == "__main__":
    main()
