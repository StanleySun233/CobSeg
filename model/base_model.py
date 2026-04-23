import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.dts_data import mean_pool
from utils.dts_utils import (
    dialogues_used_for_stream,
    evaluate_all,
    print_metrics,
    save_sample_predictions,
)


class BaseModel(nn.Module):
    default_lr = 1e-3
    default_lr_patience = 5
    default_lr_factor = 0.5
    default_min_lr = 1e-6
    default_early_stop = 10

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        object.__setattr__(self, "_runtime_main_encoder", None)
        self.loop_use_crf = False

    @property
    def main_encoder(self) -> nn.Module | None:
        return self._runtime_main_encoder

    def to_device(self):
        return self.to(self.device)

    def configure_runtime(
        self,
        *,
        main_encoder: nn.Module | None = None,
        use_crf: bool = False,
        device: torch.device | None = None,
    ):
        object.__setattr__(self, "_runtime_main_encoder", main_encoder)
        self.loop_use_crf = bool(use_crf)
        if device is not None:
            self.device = device
        return self

    def current_device(self) -> torch.device:
        try:
            return next(self.parameters()).device
        except StopIteration:
            return self.device

    def iter_trainable_parameters(self):
        for p in self.parameters():
            if p.requires_grad:
                yield p
        if self.main_encoder is not None:
            for p in self.main_encoder.parameters():
                if p.requires_grad:
                    yield p

    def count_trainable_parameters(
        self,
        *,
        include_model: bool = True,
        include_main_encoder: bool = True,
    ) -> int:
        total = 0
        if include_model:
            total += sum(p.numel() for p in self.parameters() if p.requires_grad)
        if include_main_encoder and self.main_encoder is not None:
            total += sum(p.numel() for p in self.main_encoder.parameters() if p.requires_grad)
        return total

    def make_optimizer(
        self,
        model_lr: float,
        main_encoder_lr: float | None = None,
        include_main_encoder: bool = True,
    ) -> torch.optim.Optimizer:
        model_params = [p for p in self.parameters() if p.requires_grad]
        param_groups: list[dict[str, Any]] = []
        if model_params:
            param_groups.append({"params": model_params, "lr": model_lr})
        if include_main_encoder and self.main_encoder is not None:
            encoder_lr = model_lr if main_encoder_lr is None else main_encoder_lr
            encoder_params = [p for p in self.main_encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": encoder_lr})
        if len(param_groups) <= 1:
            return torch.optim.Adam(model_params, lr=model_lr)
        return torch.optim.AdamW(param_groups)

    def save_checkpoint(self, ckpt_path: Path) -> None:
        if self.main_encoder is None:
            torch.save(self.state_dict(), ckpt_path)
            return
        payload = {
            "model_state": self.state_dict(),
            "main_encoder_state": self.main_encoder.state_dict(),
        }
        torch.save(payload, ckpt_path)

    def load_checkpoint(
        self,
        ckpt_path: Path,
        *,
        device: torch.device | None = None,
    ) -> tuple[torch.nn.modules.module._IncompatibleKeys, bool]:
        map_device = self.current_device() if device is None else device
        payload = torch.load(ckpt_path, map_location=map_device)
        encoder_loaded = False
        if isinstance(payload, dict) and "model_state" in payload:
            load_res = self.load_state_dict(payload["model_state"], strict=False)
            if self.main_encoder is not None and "main_encoder_state" in payload:
                self.main_encoder.load_state_dict(payload["main_encoder_state"], strict=False)
                encoder_loaded = True
            return load_res, encoder_loaded
        load_res = self.load_state_dict(payload, strict=False)
        return load_res, encoder_loaded

    def sync_checkpoint_state(self, load_res) -> None:
        if hasattr(self, "sync_topic_branch_from_sentence"):
            if any(
                k.startswith("lstm_t")
                or k.startswith("res_t")
                or k.startswith("head_t")
                for k in load_res.missing_keys
            ):
                self.sync_topic_branch_from_sentence()
        if hasattr(self, "sync_start_heads_from_end"):
            if any(k.startswith("head_s_start") or k.startswith("head_w_start") for k in load_res.missing_keys):
                self.sync_start_heads_from_end()

    @staticmethod
    def _cfg_to_dict(cfg: Any) -> dict[str, Any]:
        if is_dataclass(cfg):
            return asdict(cfg)
        if hasattr(cfg, "__dict__"):
            return dict(vars(cfg))
        return {"repr": repr(cfg)}

    @staticmethod
    def _format_metrics_brief(metrics: dict[str, float], prefix: str = "Val") -> str:
        return (
            f"[{prefix}] PK={metrics['PK']:.4f}  WD={metrics['WD']:.4f}  "
            f"F1={metrics['F1']:.4f}  Score={metrics['Score']:.4f}"
        )

    @staticmethod
    def _build_decay_targets(
        tags_t: torch.Tensor,
        lengths_t: torch.Tensor,
        tau: float,
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

    @staticmethod
    def _pool_pair_hidden(
        last_hidden: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if last_hidden.size(1) == 0:
            return last_hidden.new_zeros(last_hidden.size(0), last_hidden.size(-1))
        return last_hidden[:, 0, :]

    @staticmethod
    def _encode_main_batch(
        enc_model: nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, umax, tok_len = input_ids.shape
        flat_ids = input_ids.view(bsz * umax, tok_len)
        flat_mask = attention_mask.view(bsz * umax, tok_len)
        utt_mask = (
            torch.arange(umax, device=input_ids.device).unsqueeze(0)
            < lengths.to(input_ids.device).unsqueeze(1)
        ).view(-1)
        valid_ids = flat_ids[utt_mask]
        valid_mask = flat_mask[utt_mask]
        out = enc_model(input_ids=valid_ids, attention_mask=valid_mask)
        hidden = out.last_hidden_state
        sent = mean_pool(hidden, valid_mask.float())
        hidden_dim = hidden.size(-1)
        x_s = torch.zeros(
            bsz * umax,
            hidden_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        x_w = torch.zeros(
            bsz * umax,
            tok_len,
            hidden_dim,
            device=hidden.device,
            dtype=hidden.dtype,
        )
        x_s[utt_mask] = sent
        x_w[utt_mask] = hidden
        x_s = x_s.view(bsz, umax, hidden_dim)
        x_w = x_w.view(bsz, umax, tok_len, hidden_dim)
        tok_m = attention_mask.to(hidden.device, dtype=hidden.dtype)
        return x_s, x_w, tok_m

    def prepare_batch_inputs(
        self,
        batch: tuple,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        if len(batch) == 7 and batch[0].dtype != torch.long:
            x_s, x_w, tok_m, x_t, labels, lengths, kw_scores = batch
            x_s = x_s.to(device)
            x_w = x_w.to(device)
            tok_m = tok_m.to(device)
            x_t = x_t.to(device)
            labels = labels.to(device)
            kw_scores = kw_scores.to(device)
            return x_s, x_w, tok_m, x_t, labels, lengths, kw_scores, None, None

        if len(batch) == 8:
            if self.main_encoder is None:
                raise ValueError("main_encoder is required for NSP cross-encoder batches.")
            input_ids, attention_mask, pair_ids, pair_attention_mask, x_t, labels, lengths, kw_scores = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            pair_ids = pair_ids.to(device)
            pair_attention_mask = pair_attention_mask.to(device)
            x_t = x_t.to(device)
            labels = labels.to(device)
            kw_scores = kw_scores.to(device)
            x_s, x_w, tok_m = self._encode_main_batch(self.main_encoder, input_ids, attention_mask, lengths)
            return x_s, x_w, tok_m, x_t, labels, lengths, kw_scores, pair_ids, pair_attention_mask

        if len(batch) == 6:
            if self.main_encoder is None:
                raise ValueError("main_encoder is required for finetune_main_encoder batches.")
            input_ids, attention_mask, x_t, labels, lengths, kw_scores = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            x_t = x_t.to(device)
            labels = labels.to(device)
            kw_scores = kw_scores.to(device)
            x_s, x_w, tok_m = self._encode_main_batch(self.main_encoder, input_ids, attention_mask, lengths)
            return x_s, x_w, tok_m, x_t, labels, lengths, kw_scores, None, None

        raise ValueError(f"Unsupported batch format with {len(batch)} items.")

    def compute_nsp_outputs(
        self,
        pair_input_ids: torch.Tensor | None,
        pair_attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if pair_input_ids is None or pair_attention_mask is None:
            return None, None, None, None
        if self.main_encoder is None:
            raise ValueError("NSP cross-encoder inputs require a trainable main encoder.")
        if not hasattr(self, "classify_nsp_pairs"):
            raise ValueError("NSP cross-encoder path requires a model with NSP heads.")

        bsz, umax, pair_len = pair_input_ids.shape
        flat_ids = pair_input_ids.view(bsz * umax, pair_len)
        flat_mask = pair_attention_mask.view(bsz * umax, pair_len)
        pair_valid = flat_mask.sum(dim=1) > 0
        if not pair_valid.any():
            zero_mask = pair_valid.view(bsz, umax)
            return None, None, None, zero_mask

        valid_ids = flat_ids[pair_valid]
        valid_mask = flat_mask[pair_valid]
        out = self.main_encoder(input_ids=valid_ids, attention_mask=valid_mask)
        pooled = self._pool_pair_hidden(out.last_hidden_state, valid_mask)

        pair_hidden = torch.zeros(
            bsz * umax,
            pooled.size(-1),
            device=pooled.device,
            dtype=pooled.dtype,
        )
        pair_hidden[pair_valid] = pooled
        pair_hidden = pair_hidden.view(bsz, umax, pooled.size(-1))

        logits, proj_repr, probs = self.classify_nsp_pairs(pair_hidden)
        pair_mask = pair_valid.view(bsz, umax)
        proj_repr = proj_repr * pair_mask.unsqueeze(-1).to(proj_repr.dtype)
        probs = probs * pair_mask.to(probs.dtype)
        return logits, proj_repr, probs, pair_mask

    def forward_batch(
        self,
        x_s: torch.Tensor,
        x_w: torch.Tensor,
        tok_m: torch.Tensor,
        x_t: torch.Tensor,
        lengths: torch.Tensor,
        kw_scores: torch.Tensor | None = None,
        nsp_repr: torch.Tensor | None = None,
        nsp_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self(
            x_s,
            x_w,
            tok_m,
            x_t,
            lengths,
            kw_scores=kw_scores,
            nsp_repr=nsp_repr,
            nsp_probs=nsp_probs,
        )

    def forward_training_outputs(
        self,
        x_s: torch.Tensor,
        x_w: torch.Tensor,
        tok_m: torch.Tensor,
        x_t: torch.Tensor,
        lengths: torch.Tensor,
        kw_scores: torch.Tensor | None = None,
        nsp_repr: torch.Tensor | None = None,
        nsp_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        if hasattr(self, "forward_heads"):
            return self.forward_heads(
                x_s,
                x_w,
                tok_m,
                x_t,
                lengths,
                kw_scores=kw_scores,
                nsp_repr=nsp_repr,
                nsp_probs=nsp_probs,
            )
        emissions, mask = self.forward_batch(
            x_s,
            x_w,
            tok_m,
            x_t,
            lengths,
            kw_scores=kw_scores,
            nsp_repr=nsp_repr,
            nsp_probs=nsp_probs,
        )
        return emissions, None, None, None, mask

    def forward_stage1_outputs(
        self,
        x_s: torch.Tensor,
        x_w: torch.Tensor,
        tok_m: torch.Tensor,
        x_t: torch.Tensor,
        lengths: torch.Tensor,
        kw_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor], torch.Tensor]:
        if hasattr(self, "forward_representation_heads"):
            rep_emissions, rep_s, rep_w, rep_t, mask, *_ = self.forward_representation_heads(
                x_s, x_w, tok_m, x_t, lengths, kw_scores=kw_scores
            )
            return rep_emissions, [rep_s, rep_w, rep_t], mask
        emissions, _, _, _, mask = self.forward_training_outputs(
            x_s,
            x_w,
            tok_m,
            x_t,
            lengths,
            kw_scores=kw_scores,
        )
        return emissions, [], mask

    def compute_stage1_loss_components(
        self,
        batch: tuple,
        cfg: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = self.current_device()
        x_s, x_w, tok_m, x_t, labels, lengths, kw_scores, pair_ids, pair_attention_mask = self.prepare_batch_inputs(
            batch,
            device,
        )
        if getattr(cfg, "use_nsp_cross_encoder", False):
            nsp_logits, _, _, nsp_mask = self.compute_nsp_outputs(pair_ids, pair_attention_mask)
            if nsp_logits is None or nsp_mask is None or not nsp_mask.any():
                z = x_s.sum() * 0.0
                return z, {"main": z, "aux": z}
            targets = labels.long().masked_fill(~nsp_mask, -100)
            loss_main = nn.functional.cross_entropy(
                nsp_logits.view(-1, 2),
                targets.view(-1),
                ignore_index=-100,
            )
            z = loss_main.new_zeros(())
            return loss_main, {"main": loss_main, "aux": z}

        rep_emissions, aux_logits, mask = self.forward_stage1_outputs(
            x_s,
            x_w,
            tok_m,
            x_t,
            lengths,
            kw_scores=kw_scores,
        )
        tags = labels.long().masked_fill(~mask, 0)
        targets = tags.masked_fill(~mask, -100)
        loss_main = nn.functional.cross_entropy(
            rep_emissions.view(-1, 2),
            targets.view(-1),
            ignore_index=-100,
        )
        z = loss_main.new_zeros(())
        loss_aux_w = z
        stage1_aux_weight = float(getattr(cfg, "stage1_aux_weight", 0.0))
        if stage1_aux_weight > 0 and aux_logits:
            aux_terms = []
            for logits in aux_logits:
                aux_terms.append(
                    nn.functional.cross_entropy(
                        logits.view(-1, 2),
                        targets.view(-1),
                        ignore_index=-100,
                    )
                )
            loss_aux_w = stage1_aux_weight * torch.stack(aux_terms).mean()
        loss = loss_main + loss_aux_w
        return loss, {"main": loss_main, "aux": loss_aux_w}

    def compute_batch_loss_components(
        self,
        batch: tuple,
        *,
        ubiw_detach: bool,
        cfg: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        device = self.current_device()
        x_s, x_w, tok_m, x_t, labels, lengths, kw_scores, pair_ids, pair_attention_mask = self.prepare_batch_inputs(
            batch,
            device,
        )
        nsp_logits, nsp_repr, nsp_probs, nsp_mask = self.compute_nsp_outputs(
            pair_ids,
            pair_attention_mask,
        )
        emissions, end_emissions, start_emissions, coh_logits, mask = self.forward_training_outputs(
            x_s,
            x_w,
            tok_m,
            x_t,
            lengths,
            kw_scores=kw_scores,
            nsp_repr=nsp_repr,
            nsp_probs=nsp_probs,
        )
        tags = labels.long().masked_fill(~mask, 0)
        start_tags = torch.roll(tags, shifts=1, dims=1)
        start_tags[:, 0] = 1
        start_tags = start_tags.masked_fill(~mask, 0)

        if self.loop_use_crf and getattr(self, "crf", None) is not None:
            loss_main = -self.crf(emissions, tags, mask=mask, reduction="mean")
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
            loss_end_w = float(getattr(cfg, "end_loss_weight", 1.0)) * loss_end
            loss_start_w = float(getattr(cfg, "start_loss_weight", 1.0)) * loss_start
            loss = loss + loss_end_w + loss_start_w

        loss_rank_w = z
        rank_loss_weight = float(getattr(cfg, "rank_loss_weight", 0.0))
        if rank_loss_weight > 0:
            probs = torch.softmax(emissions, dim=-1)[:, :, 1]
            rank_terms = []
            rank_margin = float(getattr(cfg, "rank_margin", 0.1))
            rank_kw_gap = float(getattr(cfg, "rank_kw_gap", 0.05))
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
        coh_aux_weight = float(getattr(cfg, "coh_aux_weight", 0.0))
        if coh_aux_weight > 0 and coh_logits is not None:
            coh_target = 1.0 - tags.to(coh_logits.dtype)
            pos_idx = torch.arange(coh_logits.size(1), device=coh_logits.device).unsqueeze(0)
            next_mask = pos_idx < (lengths.to(coh_logits.device).unsqueeze(1) - 1)
            if next_mask.any():
                coh_loss = nn.functional.binary_cross_entropy_with_logits(
                    coh_logits[next_mask],
                    coh_target[next_mask],
                )
                loss_coh_w = coh_aux_weight * coh_loss
                loss = loss + loss_coh_w

        loss_nsp_w = z
        nsp_stage2_aux_weight = float(getattr(cfg, "nsp_stage2_aux_weight", 0.0))
        if nsp_stage2_aux_weight > 0 and nsp_logits is not None and nsp_mask is not None:
            nsp_targets = labels.long().masked_fill(~nsp_mask, -100)
            nsp_loss = nn.functional.cross_entropy(
                nsp_logits.view(-1, 2),
                nsp_targets.view(-1),
                ignore_index=-100,
            )
            loss_nsp_w = nsp_stage2_aux_weight * nsp_loss
            loss = loss + loss_nsp_w

        loss_ubiw_w = z
        ubiw_aux_weight = float(getattr(cfg, "ubiw_aux_weight", 0.0))
        if ubiw_aux_weight > 0 and hasattr(self, "get_ubiw_weights"):
            ubiw_aux_tau = float(getattr(cfg, "ubiw_aux_tau", 2.0))
            if hasattr(self, "get_ubiw_weights_dual"):
                dual = self.get_ubiw_weights_dual(x_s, lengths, detach=ubiw_detach)
                ubiw_end_pred, ubiw_start_pred = dual
                ubiw_end_tgt, valid_mask = self._build_decay_targets(tags, lengths, tau=ubiw_aux_tau)
                ubiw_start_tgt, _ = self._build_decay_targets(start_tags, lengths, tau=ubiw_aux_tau)
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
                ubiw_pred = self.get_ubiw_weights(x_s, lengths, detach=ubiw_detach)
                ubiw_tgt, valid_mask = self._build_decay_targets(tags, lengths, tau=ubiw_aux_tau)
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
            "nsp": loss_nsp_w,
            "ubiw": loss_ubiw_w,
        }
        return loss, parts

    def train_stage1_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        cfg: Any,
        *,
        epoch: int,
        epochs: int,
        grad_clip: float = 1.0,
    ) -> dict[str, float]:
        self.train()
        if self.main_encoder is not None:
            self.main_encoder.train()
        keys = ("loss", "main", "aux")
        acc = {k: 0.0 for k in keys}
        pbar = tqdm(loader, desc=f"stage1 {epoch}/{epochs}", leave=True, dynamic_ncols=True)

        for step, batch in enumerate(pbar, start=1):
            loss, parts = self.compute_stage1_loss_components(batch, cfg)
            optimizer.zero_grad()
            loss.backward()
            params = list(self.iter_trainable_parameters())
            if params:
                nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            acc["loss"] += loss.item()
            acc["main"] += parts["main"].item()
            acc["aux"] += parts["aux"].item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{acc['loss'] / step:.4f}")

        nbatch = max(len(loader), 1)
        return {k: acc[k] / nbatch for k in keys}

    def eval_stage1_loss_epoch(
        self,
        loader: DataLoader,
        cfg: Any,
    ) -> dict[str, float]:
        self.eval()
        if self.main_encoder is not None:
            self.main_encoder.eval()
        keys = ("loss", "main", "aux")
        acc = {k: 0.0 for k in keys}
        nbatch = max(len(loader), 1)
        with torch.no_grad():
            for batch in loader:
                loss, parts = self.compute_stage1_loss_components(batch, cfg)
                acc["loss"] += loss.item()
                acc["main"] += parts["main"].item()
                acc["aux"] += parts["aux"].item()
        return {k: acc[k] / nbatch for k in keys}

    def train_epoch(
        self,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        cfg: Any,
        *,
        epoch: int,
        epochs: int,
        grad_clip: float = 1.0,
    ) -> dict[str, float]:
        self.train()
        if self.main_encoder is not None:
            self.main_encoder.train()
        keys = ("loss", "main", "end", "start", "rank", "coh", "nsp", "ubiw")
        acc = {k: 0.0 for k in keys}
        pbar = tqdm(loader, desc=f"train {epoch}/{epochs}", leave=True, dynamic_ncols=True)

        for step, batch in enumerate(pbar, start=1):
            loss, parts = self.compute_batch_loss_components(
                batch,
                ubiw_detach=False,
                cfg=cfg,
            )
            optimizer.zero_grad()
            loss.backward()
            params = list(self.iter_trainable_parameters())
            if params:
                nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()

            acc["loss"] += loss.item()
            for name in ("main", "end", "start", "rank", "coh", "nsp", "ubiw"):
                acc[name] += parts[name].item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{acc['loss'] / step:.4f}")

        nbatch = max(len(loader), 1)
        return {k: acc[k] / nbatch for k in keys}

    def eval_epoch(
        self,
        loader: DataLoader,
        cfg: Any,
    ) -> dict[str, float]:
        self.eval()
        if self.main_encoder is not None:
            self.main_encoder.eval()
        keys = ("loss", "main", "end", "start", "rank", "coh", "nsp", "ubiw")
        acc = {k: 0.0 for k in keys}
        nbatch = max(len(loader), 1)
        with torch.no_grad():
            for batch in loader:
                loss, parts = self.compute_batch_loss_components(
                    batch,
                    ubiw_detach=True,
                    cfg=cfg,
                )
                acc["loss"] += loss.item()
                for name in ("main", "end", "start", "rank", "coh", "nsp", "ubiw"):
                    acc[name] += parts[name].item()
        return {k: acc[k] / nbatch for k in keys}

    def predict(
        self,
        loader: DataLoader,
    ) -> tuple[list[list[int]], list[list[int]]]:
        device = self.current_device()
        self.eval()
        if self.main_encoder is not None:
            self.main_encoder.eval()
        all_preds: list[list[int]] = []
        all_labels: list[list[int]] = []

        with torch.no_grad():
            for batch in loader:
                x_s, x_w, tok_m, x_t, labels, lengths, kw_scores, pair_ids, pair_attention_mask = self.prepare_batch_inputs(
                    batch,
                    device,
                )
                _, nsp_repr, nsp_probs, _ = self.compute_nsp_outputs(pair_ids, pair_attention_mask)
                emissions, mask = self.forward_batch(
                    x_s,
                    x_w,
                    tok_m,
                    x_t,
                    lengths,
                    kw_scores=kw_scores,
                    nsp_repr=nsp_repr,
                    nsp_probs=nsp_probs,
                )

                if self.loop_use_crf and getattr(self, "crf", None) is not None:
                    batch_preds = self.crf.decode(emissions, mask=mask)
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

    def predict_stage1(
        self,
        loader: DataLoader,
    ) -> tuple[list[list[int]], list[list[int]]]:
        device = self.current_device()
        self.eval()
        if self.main_encoder is not None:
            self.main_encoder.eval()
        all_preds: list[list[int]] = []
        all_labels: list[list[int]] = []

        with torch.no_grad():
            for batch in loader:
                x_s, x_w, tok_m, x_t, labels, lengths, kw_scores, pair_ids, pair_attention_mask = self.prepare_batch_inputs(
                    batch,
                    device,
                )
                if pair_ids is not None and pair_attention_mask is not None:
                    nsp_logits, _, _, nsp_mask = self.compute_nsp_outputs(pair_ids, pair_attention_mask)
                    if nsp_logits is None or nsp_mask is None or not nsp_mask.any():
                        pred_ids = torch.zeros_like(labels, dtype=torch.long)
                    else:
                        pred_ids = nsp_logits.argmax(dim=-1)
                        pred_ids = pred_ids.masked_fill(~nsp_mask, 0)
                else:
                    rep_emissions, _, mask = self.forward_stage1_outputs(
                        x_s,
                        x_w,
                        tok_m,
                        x_t,
                        lengths,
                        kw_scores=kw_scores,
                    )
                    pred_ids = rep_emissions.argmax(dim=-1)
                for i, length in enumerate(lengths.tolist()):
                    all_preds.append(pred_ids[i, :length].tolist())
                    all_labels.append(labels[i, :length].int().tolist())

        return all_preds, all_labels

    def _evaluate_checkpoint_and_save(
        self,
        *,
        ckpt_path: Path,
        val_loader: DataLoader | None,
        test_loader: DataLoader,
        test_dialogues: list,
        max_utterances: int,
        num_samples: int,
        seed: int,
        results_path: Path | None = None,
        cfg: Any | None = None,
        stage1_info: dict[str, Any] | None = None,
        include_results_json: bool = True,
    ) -> tuple[dict[str, float] | None, dict[str, float]]:
        print(f"Loading checkpoint: {ckpt_path}")
        load_res, encoder_loaded = self.load_checkpoint(ckpt_path)
        self.sync_checkpoint_state(load_res)
        if self.main_encoder is not None and not encoder_loaded:
            print("[warn] checkpoint does not contain main encoder weights; using base encoder init.")

        metrics_val = None
        if val_loader is not None:
            preds_v, labels_v = self.predict(val_loader)
            metrics_val = evaluate_all(preds_v, labels_v)
            print_metrics(metrics_val, prefix="Val")

        preds, labels = self.predict(test_loader)
        metrics_test = evaluate_all(preds, labels)
        print_metrics(metrics_test, prefix="Test")

        if include_results_json and results_path is not None:
            payload = {
                "config": self._cfg_to_dict(cfg),
                "selection_metric": "PK",
                "selection_mode": "min",
                "metrics_val": metrics_val,
                "metrics_test": metrics_test,
                "nsp_cross_encoder": bool(getattr(cfg, "use_nsp_cross_encoder", False)),
            }
            if stage1_info is not None:
                payload["stage1"] = stage1_info
            with open(results_path, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Results saved to {results_path}")

        save_sample_predictions(
            dialogues_used_for_stream(test_dialogues, max_utterances),
            preds,
            labels,
            out_path=ckpt_path.parent / "sample_predictions.csv",
            n=num_samples,
            seed=seed,
        )
        return metrics_val, metrics_test

    def run_experiment(
        self,
        *,
        cfg: Any,
        train_loader: DataLoader,
        val_loader: DataLoader | None,
        test_loader: DataLoader,
        ckpt_dir: Path,
        test_dialogues: list,
        max_utterances: int,
    ) -> None:
        if getattr(cfg, "eval_only", False):
            _, _ = self._evaluate_checkpoint_and_save(
                ckpt_path=ckpt_dir / "best.pt",
                val_loader=val_loader,
                test_loader=test_loader,
                test_dialogues=test_dialogues,
                max_utterances=max_utterances,
                num_samples=int(getattr(cfg, "num_samples", -1)),
                seed=int(getattr(cfg, "seed", 42)),
                include_results_json=False,
            )
            return

        if val_loader is None:
            raise SystemExit("No valid/val/dev split found in dataset.")

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best.pt"
        stage1_ckpt_path = ckpt_dir / "stage1_best.pt"
        results_path = ckpt_dir / "results.json"

        stage1_info = None
        if getattr(cfg, "two_stage_training", False) and int(getattr(cfg, "stage1_epochs", 0)) > 0:
            if getattr(cfg, "use_nsp_cross_encoder", False):
                print("\n--- Stage 1: RoBERTa pair cross-encoder warmup ---")
            else:
                print("\n--- Stage 1: transition-aware representation warmup ---")
            stage1_optimizer = self.make_optimizer(
                float(getattr(cfg, "stage1_lr", self.default_lr)),
                main_encoder_lr=float(getattr(cfg, "stage1_main_encoder_lr", getattr(cfg, "stage1_lr", self.default_lr))),
            )
            stage1_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                stage1_optimizer,
                mode="min",
                factor=float(getattr(cfg, "lr_factor", self.default_lr_factor)),
                patience=int(getattr(cfg, "lr_patience", self.default_lr_patience)),
                min_lr=float(getattr(cfg, "min_lr", self.default_min_lr)),
            )
            best_stage1_pk = float("inf")
            best_stage1_score = float("-inf")
            stage1_no_improve = 0
            stage1_epochs = int(getattr(cfg, "stage1_epochs", 0))

            for epoch in range(1, stage1_epochs + 1):
                train_stats = self.train_stage1_epoch(
                    train_loader,
                    stage1_optimizer,
                    cfg,
                    epoch=epoch,
                    epochs=stage1_epochs,
                )
                val_loss_stats = self.eval_stage1_loss_epoch(
                    val_loader,
                    cfg,
                )
                preds_v, labels_v = self.predict_stage1(val_loader)
                metrics_val = evaluate_all(preds_v, labels_v)
                stage1_scheduler.step(metrics_val["PK"])

                lr_msg = f"{stage1_optimizer.param_groups[0]['lr']:.2e}"
                if len(stage1_optimizer.param_groups) > 1:
                    extra_lrs = [f"{group['lr']:.2e}" for group in stage1_optimizer.param_groups[1:]]
                    lr_msg = f"{lr_msg}/" + "/".join(extra_lrs)
                print(
                    f"Stage1 {epoch:3d}/{stage1_epochs}  "
                    f"tr={train_stats['loss']:.4f}  "
                    f"val={val_loss_stats['loss']:.4f}  "
                    f"lr={lr_msg}  "
                    f"{self._format_metrics_brief(metrics_val, prefix='Val')}"
                )

                current_pk = float(metrics_val["PK"])
                current_score = float(metrics_val["Score"])
                if (current_pk < best_stage1_pk) or (
                    current_pk == best_stage1_pk and current_score > best_stage1_score
                ):
                    best_stage1_pk = current_pk
                    best_stage1_score = current_score
                    self.save_checkpoint(stage1_ckpt_path)
                    print(
                        "  ↳ Saved stage1 checkpoint  "
                        f"(Val PK={best_stage1_pk:.4f}, Score={best_stage1_score:.4f})"
                    )
                    stage1_no_improve = 0
                else:
                    stage1_no_improve += 1

                if int(getattr(cfg, "early_stop", 0)) > 0 and stage1_no_improve >= int(getattr(cfg, "early_stop", 0)):
                    print(
                        f"Stage1 early stopping after {stage1_no_improve} epoch(s) "
                        f"without Val PK improvement (patience={int(getattr(cfg, 'early_stop', 0))})."
                    )
                    break

            load_res, encoder_loaded = self.load_checkpoint(stage1_ckpt_path)
            self.sync_checkpoint_state(load_res)
            if self.main_encoder is not None and not encoder_loaded:
                print("[warn] stage1 checkpoint does not contain main encoder weights; using base encoder init.")
            stage1_info = {
                "mode": "nsp_cross_encoder" if getattr(cfg, "use_nsp_cross_encoder", False) else "transition_repr",
                "best_pk": best_stage1_pk,
                "best_score": best_stage1_score,
                "epochs": stage1_epochs,
                "ckpt": str(stage1_ckpt_path),
            }
            print(
                f"Stage1 loaded best checkpoint from {stage1_ckpt_path}  "
                f"(Val PK={best_stage1_pk:.4f})"
            )

        optimizer = self.make_optimizer(
            float(getattr(cfg, "lr", self.default_lr)),
            main_encoder_lr=float(getattr(cfg, "main_encoder_lr", getattr(cfg, "lr", self.default_lr))),
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(getattr(cfg, "lr_factor", self.default_lr_factor)),
            patience=int(getattr(cfg, "lr_patience", self.default_lr_patience)),
            min_lr=float(getattr(cfg, "min_lr", self.default_min_lr)),
        )

        best_pk = float("inf")
        best_score_at_best_pk = float("-inf")
        epochs_no_improve = 0
        train_epochs = int(getattr(cfg, "epochs", 0))

        for epoch in range(1, train_epochs + 1):
            train_stats = self.train_epoch(
                train_loader,
                optimizer,
                cfg,
                epoch=epoch,
                epochs=train_epochs,
            )
            val_loss_stats = self.eval_epoch(
                val_loader,
                cfg,
            )
            preds_v, labels_v = self.predict(val_loader)
            metrics_val = evaluate_all(preds_v, labels_v)
            scheduler.step(metrics_val["PK"])

            lr_msg = f"{optimizer.param_groups[0]['lr']:.2e}"
            if len(optimizer.param_groups) > 1:
                extra_lrs = [f"{group['lr']:.2e}" for group in optimizer.param_groups[1:]]
                lr_msg = f"{lr_msg}/" + "/".join(extra_lrs)

            print(
                f"Epoch {epoch:3d}/{train_epochs}  "
                f"tr={train_stats['loss']:.4f}  "
                f"val={val_loss_stats['loss']:.4f}  "
                f"lr={lr_msg}  "
                f"{self._format_metrics_brief(metrics_val, prefix='Val')}"
            )

            current_pk = float(metrics_val["PK"])
            current_score = float(metrics_val["Score"])
            if (current_pk < best_pk) or (
                current_pk == best_pk and current_score > best_score_at_best_pk
            ):
                best_pk = current_pk
                best_score_at_best_pk = current_score
                self.save_checkpoint(ckpt_path)
                print(
                    "  ↳ Saved best checkpoint  "
                    f"(Val PK={best_pk:.4f}, Score={best_score_at_best_pk:.4f})"
                )
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            if int(getattr(cfg, "early_stop", 0)) > 0 and epochs_no_improve >= int(getattr(cfg, "early_stop", 0)):
                print(
                    f"Early stopping after {epochs_no_improve} epoch(s) "
                    f"without Val PK improvement (patience={int(getattr(cfg, 'early_stop', 0))})."
                )
                break

        print("\n--- Final evaluation (best checkpoint) ---")
        self._evaluate_checkpoint_and_save(
            ckpt_path=ckpt_path,
            val_loader=val_loader,
            test_loader=test_loader,
            test_dialogues=test_dialogues,
            max_utterances=max_utterances,
            num_samples=int(getattr(cfg, "num_samples", -1)),
            seed=int(getattr(cfg, "seed", 42)),
            results_path=results_path,
            cfg=cfg,
            stage1_info=stage1_info,
            include_results_json=True,
        )
