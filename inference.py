import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from model.base_model import BaseModel
from model.dud import DUD
from model.bert_dts import PureBertSegmenter
from utils.dialogue_dataset import DialogueDataset
from utils.dts_data import (
    EmbeddedDialogueDataset,
    MAX_UTT_TOKENS,
    MAX_UTTERANCES,
    collate_fn,
    topic_channel_sets_from_info,
)
from utils.dts_utils import (
    dialogues_used_for_stream,
    evaluate_all,
    print_metrics,
    run_checkpoint_dir,
    save_sample_predictions,
)
from utils.utils import resolve_dataset_path

CS_ENCODER_NAME = "princeton-nlp/sup-simcse-bert-base-uncased"


@dataclass
class ExperimentConfig:
    model_name: str
    dataset: str
    encoder: str
    emb_batch: int
    epochs: int
    batch_size: int
    seed: int
    lr: float
    lr_patience: int
    lr_factor: float
    min_lr: float
    early_stop: int
    num_samples: int
    eval_only: bool
    exp_name: str
    stream_mode: str
    disable_token_transformer: bool
    disable_crf: bool
    disable_ubiw: bool
    topic_json_path: str
    rank_loss_weight: float
    rank_margin: float
    rank_kw_gap: float
    end_loss_weight: float
    start_loss_weight: float
    edge_gate_alpha: float
    edge_gate_gamma: float
    ubiw_aux_weight: float
    ubiw_aux_tau: float
    coh_aux_weight: float


_DATASET_CFG_CACHE: dict = {}


def _load_dataset_config(config_path: str) -> dict:
    """Load dataset_config.yaml once and cache it."""
    global _DATASET_CFG_CACHE
    if config_path not in _DATASET_CFG_CACHE:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                _DATASET_CFG_CACHE[config_path] = yaml.safe_load(f) or {}
        else:
            print(f"[warn] dataset_config not found: {config_path}  (using built-in defaults)")
            _DATASET_CFG_CACHE[config_path] = {}
    return _DATASET_CFG_CACHE[config_path]


_DATASET_CONFIG_PATH = Path(__file__).resolve().parent / "data" / "dataset_config.yaml"


def resolve_dataset_limits(dataset: str) -> tuple[int, int]:
    """Return (max_utt_tokens, max_utterances) from data/dataset_config.yaml."""
    ds_stem = Path(resolve_dataset_path(dataset)).stem
    ds_cfg = _load_dataset_config(str(_DATASET_CONFIG_PATH)).get(ds_stem, {})
    return (
        int(ds_cfg.get("max_utt_tokens", MAX_UTT_TOKENS)),
        int(ds_cfg.get("max_utterances",  MAX_UTTERANCES)),
    )


def compute_pos_weight(dataset: EmbeddedDialogueDataset) -> float:
    total_neg, total_pos = 0, 0
    for _, _, _, _, labels, _ in dataset.samples:
        pos = labels.sum().item()
        neg = len(labels) - pos
        total_pos += pos
        total_neg += neg
    if total_pos == 0:
        return 1.0
    return total_neg / total_pos


def _build_decay_targets(
    tags_t: torch.Tensor, lengths_t: torch.Tensor, tau: float
) -> tuple[torch.Tensor, torch.Tensor]:
    bsz, tmax = tags_t.shape
    target = torch.zeros((bsz, tmax), dtype=torch.float32, device=tags_t.device)
    mask_t = torch.zeros((bsz, tmax), dtype=torch.bool, device=tags_t.device)
    pos_idx = torch.arange(tmax, device=tags_t.device, dtype=torch.float32)
    for bi, lv in enumerate(lengths_t.tolist()):
        l = int(lv)
        if l <= 0:
            continue
        mask_t[bi, :l] = True
        b = torch.nonzero(tags_t[bi, :l] > 0, as_tuple=False).squeeze(-1)
        if b.numel() == 0:
            continue
        d = (pos_idx[:l].unsqueeze(1) - b.float().unsqueeze(0)).abs().amin(dim=1)
        target[bi, :l] = torch.exp(-d / max(float(tau), 1e-6))
    return target, mask_t


def compute_batch_loss_components(
    model: nn.Module,
    batch: tuple,
    device: torch.device,
    use_crf: bool,
    rank_loss_weight: float,
    rank_margin: float,
    rank_kw_gap: float,
    end_loss_weight: float,
    start_loss_weight: float,
    ubiw_aux_weight: float,
    ubiw_aux_tau: float,
    coh_aux_weight: float,
    ubiw_detach: bool,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x_s, x_w, tok_m, x_t, labels, lengths, kw_scores = batch
    x_s = x_s.to(device)
    x_w = x_w.to(device)
    tok_m = tok_m.to(device)
    x_t = x_t.to(device)
    labels = labels.to(device)
    kw_scores = kw_scores.to(device)

    if hasattr(model, "forward_heads"):
        emissions, end_emissions, start_emissions, coh_logits, mask = model.forward_heads(
            x_s, x_w, tok_m, x_t, lengths, kw_scores=kw_scores
        )
    else:
        emissions, mask = model(x_s, x_w, tok_m, x_t, lengths, kw_scores=kw_scores)
        end_emissions = None
        start_emissions = None
        coh_logits = None
    tags = labels.long().masked_fill(~mask, 0)
    start_tags = torch.roll(tags, shifts=1, dims=1)
    start_tags[:, 0] = 1
    start_tags = start_tags.masked_fill(~mask, 0)

    if use_crf:
        loss_main = -model.crf(emissions, tags, mask=mask, reduction="mean")
    else:
        targets = tags.masked_fill(~mask, -100)
        loss_main = nn.functional.cross_entropy(
            emissions.view(-1, 2),
            targets.view(-1),
            ignore_index=-100,
        )

    z = loss_main.new_zeros(())
    loss_end_w = z
    loss_start_w = z
    loss = loss_main

    if end_emissions is not None and start_emissions is not None:
        end_targets = tags.masked_fill(~mask, -100)
        start_targets = start_tags.masked_fill(~mask, -100)
        loss_end = nn.functional.cross_entropy(
            end_emissions.view(-1, 2),
            end_targets.view(-1),
            ignore_index=-100,
        )
        loss_start = nn.functional.cross_entropy(
            start_emissions.view(-1, 2),
            start_targets.view(-1),
            ignore_index=-100,
        )
        loss_end_w = end_loss_weight * loss_end
        loss_start_w = start_loss_weight * loss_start
        loss = loss + loss_end_w + loss_start_w

    loss_rank_w = z
    if rank_loss_weight > 0:
        probs = torch.softmax(emissions, dim=-1)[:, :, 1]
        rank_terms = []
        for i, length in enumerate(lengths.tolist()):
            l = int(length)
            if l <= 1:
                continue
            p = probs[i, :l]
            k_slice = kw_scores[i, :l]
            if k_slice.dim() == 2:
                k = k_slice.sum(dim=-1)
            else:
                k = k_slice
            p_diff = p.unsqueeze(1) - p.unsqueeze(0)
            k_diff = k.unsqueeze(1) - k.unsqueeze(0)
            valid = k_diff > rank_kw_gap
            if valid.any():
                rank_terms.append(torch.relu(rank_margin - p_diff[valid]).mean())
        if rank_terms:
            rank_loss = torch.stack(rank_terms).mean()
            loss_rank_w = rank_loss_weight * rank_loss
            loss = loss + loss_rank_w

    loss_coh_w = z
    if coh_aux_weight > 0 and coh_logits is not None:
        coh_target = (1.0 - tags.to(coh_logits.dtype))
        pos_idx = torch.arange(coh_logits.size(1), device=coh_logits.device).unsqueeze(0)
        next_mask = pos_idx < (lengths.to(coh_logits.device).unsqueeze(1) - 1)
        if next_mask.any():
            coh_loss = nn.functional.binary_cross_entropy_with_logits(
                coh_logits[next_mask],
                coh_target[next_mask],
            )
            loss_coh_w = coh_aux_weight * coh_loss
            loss = loss + loss_coh_w

    loss_ubiw_w = z
    if ubiw_aux_weight > 0 and hasattr(model, "get_ubiw_weights"):
        if hasattr(model, "get_ubiw_weights_dual"):
            dual = model.get_ubiw_weights_dual(x_s, lengths, detach=ubiw_detach)
            ubiw_end_pred, ubiw_start_pred = dual
            ubiw_end_tgt, valid_mask = _build_decay_targets(tags, lengths, tau=ubiw_aux_tau)
            ubiw_start_tgt, _ = _build_decay_targets(start_tags, lengths, tau=ubiw_aux_tau)
            if valid_mask.any():
                ubiw_end_loss = nn.functional.mse_loss(
                    ubiw_end_pred[valid_mask], ubiw_end_tgt[valid_mask]
                )
                ubiw_start_loss = nn.functional.mse_loss(
                    ubiw_start_pred[valid_mask], ubiw_start_tgt[valid_mask]
                )
                loss_ubiw_w = ubiw_aux_weight * 0.5 * (ubiw_end_loss + ubiw_start_loss)
                loss = loss + loss_ubiw_w
        else:
            ubiw_pred = model.get_ubiw_weights(x_s, lengths, detach=ubiw_detach)
            ubiw_tgt, valid_mask = _build_decay_targets(tags, lengths, tau=ubiw_aux_tau)
            if valid_mask.any():
                ubiw_loss = nn.functional.mse_loss(ubiw_pred[valid_mask], ubiw_tgt[valid_mask])
                loss_ubiw_w = ubiw_aux_weight * ubiw_loss
                loss = loss + loss_ubiw_w

    parts = {
        "main": loss_main,
        "end": loss_end_w,
        "start": loss_start_w,
        "rank": loss_rank_w,
        "coh": loss_coh_w,
        "ubiw": loss_ubiw_w,
    }
    return loss, parts


def _format_metrics_brief(metrics: dict[str, float], prefix: str = "Val") -> str:
    return (
        f"[{prefix}] PK={metrics['PK']:.4f}  WD={metrics['WD']:.4f}  "
        f"F1={metrics['F1']:.4f}  Score={metrics['Score']:.4f}"
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_crf: bool,
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.1,
    rank_kw_gap: float = 0.05,
    end_loss_weight: float = 1.0,
    start_loss_weight: float = 1.0,
    grad_clip: float = 1.0,
    ubiw_aux_weight: float = 0.0,
    ubiw_aux_tau: float = 2.0,
    coh_aux_weight: float = 0.0,
    epoch: int = 1,
    epochs: int = 1,
) -> dict[str, float]:
    model.train()
    keys = ("loss", "main", "end", "start", "rank", "coh", "ubiw")
    acc = {k: 0.0 for k in keys}
    pbar = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=True, dynamic_ncols=True)

    for step, batch in enumerate(pbar, start=1):
        loss, parts = compute_batch_loss_components(
            model,
            batch,
            device,
            use_crf=use_crf,
            rank_loss_weight=rank_loss_weight,
            rank_margin=rank_margin,
            rank_kw_gap=rank_kw_gap,
            end_loss_weight=end_loss_weight,
            start_loss_weight=start_loss_weight,
            ubiw_aux_weight=ubiw_aux_weight,
            ubiw_aux_tau=ubiw_aux_tau,
            coh_aux_weight=coh_aux_weight,
            ubiw_detach=False,
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        acc["loss"] += loss.item()
        for name in ("main", "end", "start", "rank", "coh", "ubiw"):
            acc[name] += parts[name].item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{acc['loss'] / step:.4f}")

    nbatch = max(len(loader), 1)
    return {k: acc[k] / nbatch for k in keys}


def eval_loss_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_crf: bool,
    rank_loss_weight: float = 0.0,
    rank_margin: float = 0.1,
    rank_kw_gap: float = 0.05,
    end_loss_weight: float = 1.0,
    start_loss_weight: float = 1.0,
    ubiw_aux_weight: float = 0.0,
    ubiw_aux_tau: float = 2.0,
    coh_aux_weight: float = 0.0,
) -> dict[str, float]:
    model.eval()
    keys = ("loss", "main", "end", "start", "rank", "coh", "ubiw")
    acc = {k: 0.0 for k in keys}
    nbatch = max(len(loader), 1)
    with torch.no_grad():
        for batch in loader:
            loss, parts = compute_batch_loss_components(
                model,
                batch,
                device,
                use_crf=use_crf,
                rank_loss_weight=rank_loss_weight,
                rank_margin=rank_margin,
                rank_kw_gap=rank_kw_gap,
                end_loss_weight=end_loss_weight,
                start_loss_weight=start_loss_weight,
                ubiw_aux_weight=ubiw_aux_weight,
                ubiw_aux_tau=ubiw_aux_tau,
                coh_aux_weight=coh_aux_weight,
                ubiw_detach=True,
            )
            acc["loss"] += loss.item()
            for name in ("main", "end", "start", "rank", "coh", "ubiw"):
                acc[name] += parts[name].item()
    return {k: acc[k] / nbatch for k in keys}


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[int]], list[list[int]]]:
    model.eval()
    all_preds: list[list[int]] = []
    all_labels: list[list[int]] = []

    with torch.no_grad():
        for x_s, x_w, tok_m, x_t, labels, lengths, kw_scores in loader:
            x_s = x_s.to(device)
            x_w = x_w.to(device)
            tok_m = tok_m.to(device)
            x_t = x_t.to(device)
            kw_scores = kw_scores.to(device)
            emissions, mask = model(x_s, x_w, tok_m, x_t, lengths, kw_scores=kw_scores)

            if use_crf:
                batch_preds = model.crf.decode(emissions, mask=mask)
            else:
                pred_ids = emissions.argmax(dim=-1)
                batch_preds = []
                for i, length in enumerate(lengths.tolist()):
                    batch_preds.append(pred_ids[i, :length].tolist())

            for i, length in enumerate(lengths.tolist()):
                pred = batch_preds[i][:length]
                true = labels[i, :length].int().tolist()
                all_preds.append(pred)
                all_labels.append(true)

    return all_preds, all_labels


def build_model(cfg: ExperimentConfig, input_dim: int, max_utt_tokens: int) -> tuple[nn.Module, bool]:
    if cfg.model_name == "dud":
        model = DUD(
            input_dim=input_dim,
            max_utt_tokens=max_utt_tokens,
            stream_mode=cfg.stream_mode,
            use_token_transformer=not cfg.disable_token_transformer,
            use_crf=not cfg.disable_crf,
            use_ubiw=not cfg.disable_ubiw,
            edge_gate_alpha=cfg.edge_gate_alpha,
            edge_gate_gamma=cfg.edge_gate_gamma,
            topic_json_path=cfg.topic_json_path,
        )
        return model, not cfg.disable_crf

    if cfg.model_name == "bert":
        model = PureBertSegmenter(input_dim=input_dim)
        return model, False

    raise ValueError(f"Unsupported model_name: {cfg.model_name}")


def _load_hf_encoder(model_name: str, device: torch.device):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
        enc_model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
        ).to(device).eval()
        return tokenizer, enc_model
    except OSError as exc:
        raise RuntimeError(
            f"Failed to load encoder '{model_name}'. "
            "The loader uses local cache when available and otherwise tries to "
            "download from Hugging Face."
        ) from exc


def run_single(cfg: ExperimentConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── resolve per-dataset token / utterance limits from dataset_config.yaml ──
    max_utt_tokens, max_utterances = resolve_dataset_limits(cfg.dataset)
    print(f"Limits  max_utt_tokens={max_utt_tokens}  max_utterances={max_utterances}"
          f"  (data/dataset_config.yaml)")

    os.environ.setdefault("HF_HUB_READ_TIMEOUT", "120")

    ds_path = resolve_dataset_path(cfg.dataset)
    print(f"Loading dataset: {ds_path}")
    ckpt_dir = run_checkpoint_dir(__file__, ds_path, cfg.exp_name)
    print(f"Run checkpoint directory: {ckpt_dir}")

    full_dataset = DialogueDataset(ds_path)
    train_dialogues = [d for d in full_dataset if d.set == "train"]
    val_dialogues = [d for d in full_dataset if d.set in ("valid", "val", "dev")]
    test_dialogues = [d for d in full_dataset if d.set == "test"]
    print(f"Train: {len(train_dialogues)}  Valid: {len(val_dialogues)}  Test: {len(test_dialogues)}")

    print(f"Loading main encoder: {cfg.encoder}")
    tokenizer, enc_model = _load_hf_encoder(cfg.encoder, device)
    print(f"Loading CS encoder: {CS_ENCODER_NAME}")
    cs_tokenizer, cs_enc_model = _load_hf_encoder(CS_ENCODER_NAME, device)
    topic_channels_by_ds: dict[str, dict[str, set[str]]] = {}
    if os.path.exists(cfg.topic_json_path):
        with open(cfg.topic_json_path, "r", encoding="utf-8") as f:
            topic_data = json.load(f)
        topic_channels_by_ds = {
            ds: topic_channel_sets_from_info(info)
            for ds, info in topic_data.items()
        }
        print(f"Loaded topic keywords from {cfg.topic_json_path}")
    topic_channels = topic_channels_by_ds.get(
        cfg.dataset, topic_channel_sets_from_info({})
    )

    train_data = EmbeddedDialogueDataset(
        train_dialogues,
        enc_model,
        tokenizer,
        device,
        batch_size=cfg.emb_batch,
        max_utterances=max_utterances,
        max_utt_tokens=max_utt_tokens,
        dataset_name=cfg.dataset,
        topic_channels=topic_channels,
        cs_enc_model=cs_enc_model,
        cs_tokenizer=cs_tokenizer,
    )
    test_data = EmbeddedDialogueDataset(
        test_dialogues,
        enc_model,
        tokenizer,
        device,
        batch_size=cfg.emb_batch,
        max_utterances=max_utterances,
        max_utt_tokens=max_utt_tokens,
        dataset_name=cfg.dataset,
        topic_channels=topic_channels,
        cs_enc_model=cs_enc_model,
        cs_tokenizer=cs_tokenizer,
    )

    if not cfg.eval_only and len(train_data.samples) == 0:
        raise SystemExit("No training dialogues after encoding.")

    ref_data = train_data if train_data.samples else test_data
    if not ref_data.samples:
        raise SystemExit("No dialogues after encoding.")

    input_dim = ref_data.samples[0][0].shape[-1]
    token_len = ref_data.samples[0][1].shape[1]
    if token_len != max_utt_tokens:
        raise ValueError(f"token length {token_len} != max_utt_tokens {max_utt_tokens}")

    train_loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, max_utterances=max_utterances),
    )
    test_loader = DataLoader(
        test_data,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, max_utterances=max_utterances),
    )

    val_loader = None
    if val_dialogues:
        val_data = EmbeddedDialogueDataset(
            val_dialogues,
            enc_model,
            tokenizer,
            device,
            batch_size=cfg.emb_batch,
            max_utterances=max_utterances,
            max_utt_tokens=max_utt_tokens,
            dataset_name=cfg.dataset,
            topic_channels=topic_channels,
            cs_enc_model=cs_enc_model,
            cs_tokenizer=cs_tokenizer,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=cfg.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, max_utterances=max_utterances),
        )

    model, use_crf = build_model(cfg, input_dim=input_dim, max_utt_tokens=max_utt_tokens)
    model = model.to(device)

    if hasattr(model, "set_class_balance"):
        pos_w = compute_pos_weight(train_data) if train_data.samples else 1.0
        model.set_class_balance(pos_w)
        print(f"Train token pos_weight (neg/pos): {pos_w:.4f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "best.pt"

    if cfg.eval_only:
        print(f"Loading checkpoint: {ckpt_path}")
        load_res = model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
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
        if val_loader is not None:
            preds_v, labels_v = predict(model, val_loader, device, use_crf=use_crf)
            print_metrics(evaluate_all(preds_v, labels_v), prefix="Val")
        preds, labels = predict(model, test_loader, device, use_crf=use_crf)
        print_metrics(evaluate_all(preds, labels), prefix="Test")
        save_sample_predictions(
            dialogues_used_for_stream(test_dialogues, max_utterances),
            preds,
            labels,
            out_path=ckpt_dir / "sample_predictions.csv",
            n=cfg.num_samples,
            seed=cfg.seed,
        )
        return

    if val_loader is None:
        raise SystemExit("No valid/val/dev split found in dataset.")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=cfg.lr_factor,
        patience=cfg.lr_patience,
        min_lr=cfg.min_lr,
    )

    best_score = float("-inf")
    epochs_no_improve = 0

    for epoch in range(1, cfg.epochs + 1):
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_crf=use_crf,
            rank_loss_weight=cfg.rank_loss_weight,
            rank_margin=cfg.rank_margin,
            rank_kw_gap=cfg.rank_kw_gap,
            end_loss_weight=cfg.end_loss_weight,
            start_loss_weight=cfg.start_loss_weight,
            ubiw_aux_weight=cfg.ubiw_aux_weight,
            ubiw_aux_tau=cfg.ubiw_aux_tau,
            coh_aux_weight=cfg.coh_aux_weight,
            epoch=epoch,
            epochs=cfg.epochs,
        )
        val_loss_stats = eval_loss_one_epoch(
            model,
            val_loader,
            device,
            use_crf=use_crf,
            rank_loss_weight=cfg.rank_loss_weight,
            rank_margin=cfg.rank_margin,
            rank_kw_gap=cfg.rank_kw_gap,
            end_loss_weight=cfg.end_loss_weight,
            start_loss_weight=cfg.start_loss_weight,
            ubiw_aux_weight=cfg.ubiw_aux_weight,
            ubiw_aux_tau=cfg.ubiw_aux_tau,
            coh_aux_weight=cfg.coh_aux_weight,
        )

        preds_v, labels_v = predict(model, val_loader, device, use_crf=use_crf)
        metrics_val = evaluate_all(preds_v, labels_v)
        scheduler.step(metrics_val["PK"])

        print(
            f"Epoch {epoch:3d}/{cfg.epochs}  "
            f"tr={train_stats['loss']:.4f}  "
            f"val={val_loss_stats['loss']:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}  "
            f"{_format_metrics_brief(metrics_val, prefix='Val')}"
        )

        if metrics_val["Score"] > best_score:
            best_score = metrics_val["Score"]
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ↳ Saved best checkpoint  (Val Score={best_score:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if cfg.early_stop > 0 and epochs_no_improve >= cfg.early_stop:
            print(
                f"Early stopping after {epochs_no_improve} epoch(s) "
                f"without Val Score improvement (patience={cfg.early_stop})."
            )
            break

    print("\n--- Final evaluation (best checkpoint) ---")
    load_res = model.load_state_dict(torch.load(ckpt_path, map_location=device), strict=False)
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
    preds_v, labels_v = predict(model, val_loader, device, use_crf=use_crf)
    metrics_val = evaluate_all(preds_v, labels_v)
    print_metrics(metrics_val, prefix="Val")
    preds, labels = predict(model, test_loader, device, use_crf=use_crf)
    metrics_test = evaluate_all(preds, labels)
    print_metrics(metrics_test, prefix="Test")

    results_path = ckpt_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"config": asdict(cfg), "metrics_val": metrics_val, "metrics_test": metrics_test}, f, indent=2)
    print(f"Results saved to {results_path}")

    save_sample_predictions(
        dialogues_used_for_stream(test_dialogues, max_utterances),
        preds,
        labels,
        out_path=ckpt_dir / "sample_predictions.csv",
        n=cfg.num_samples,
        seed=cfg.seed,
    )


def main():
    parser = argparse.ArgumentParser(description="DTS training entry (unified)")
    parser.add_argument("--model_name", default="dud", choices=("dud", "bert"))
    parser.add_argument(
        "--dataset",
        default="vhf",
        help="vhf | dialseg711 | doc2dial | tiage | superseg, or path to a .json dialogue file",
    )
    parser.add_argument("--encoder", default="BAAI/bge-m3")
    parser.add_argument("--emb_batch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=BaseModel.default_lr)
    parser.add_argument("--lr_patience", type=int, default=BaseModel.default_lr_patience)
    parser.add_argument("--lr_factor", type=float, default=BaseModel.default_lr_factor)
    parser.add_argument("--min_lr", type=float, default=BaseModel.default_min_lr)
    parser.add_argument("--early_stop", type=int, default=BaseModel.default_early_stop)
    parser.add_argument("--num_samples", type=int, default=-1)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--exp_name", type=str, default="baseline")

    parser.add_argument("--stream_mode", choices=("dual", "sentence", "token"), default="dual")
    parser.add_argument("--disable_token_transformer", action="store_true")
    parser.add_argument("--disable_crf", action="store_true")
    parser.add_argument("--disable_ubiw", action="store_true")
    parser.add_argument("--topic_json_path", type=str, default="./data/topic/topic_keywords.json")
    parser.add_argument("--rank_loss_weight", type=float, default=0.2)
    parser.add_argument("--rank_margin", type=float, default=0.1)
    parser.add_argument("--rank_kw_gap", type=float, default=0.05)
    parser.add_argument("--end_loss_weight", type=float, default=1.0)
    parser.add_argument("--start_loss_weight", type=float, default=1.0)
    parser.add_argument("--edge_gate_alpha", type=float, default=0.25)
    parser.add_argument("--edge_gate_gamma", type=float, default=1.5)
    parser.add_argument("--ubiw_aux_weight", type=float, default=0.2)
    parser.add_argument("--ubiw_aux_tau", type=float, default=2.0)
    parser.add_argument("--coh_aux_weight", type=float, default=0.2)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ExperimentConfig(**vars(args))
    run_single(cfg)


if __name__ == "__main__":
    main()
