import math
import json
import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcrf import CRF

from model.base_model import BaseModel
from utils.dts_data import topic_channel_sets_from_info
"""
Dual-stream DTS model definition.
Only model structure is provided here (no training / CLI / data pipeline).
"""


def _make_mlp_head(hidden2: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden2, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 2),
    )


class DualStreamSegmenter(BaseModel):
    def __init__(
        self,
        input_dim: int,
        max_utt_tokens: int = 64,
        stream_mode: str = "dual",
        use_token_transformer: bool = True,
        use_crf: bool = True,
        use_ubiw: bool = True,
        edge_gate_alpha: float = 0.25,
        edge_gate_gamma: float = 1.5,
        topic_json_path: str = "./data/topic/topic_keywords.json",
    ):
        super().__init__()
        self.max_utt_tokens = int(max_utt_tokens)
        self.stream_mode = stream_mode
        self.use_token_transformer = use_token_transformer
        self.use_crf = use_crf
        self.use_ubiw = use_ubiw
        self.edge_gate_alpha = float(edge_gate_alpha)
        self.edge_gate_gamma = float(edge_gate_gamma)
        hidden_dim = 256
        token_hidden = 128
        num_layers = 2
        dropout = 0.5
        nhead = 8
        while nhead > 1 and input_dim % nhead != 0:
            nhead //= 2
        tf_layers = 2
        tf_ff = max(input_dim * 4, 512)
        lstm_kw = dict(
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        h2 = hidden_dim * 2
        tf_enc = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            dim_feedforward=tf_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.token_transformer = nn.TransformerEncoder(tf_enc, num_layers=tf_layers)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.max_utt_tokens, input_dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.token_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=token_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.word_utt_proj = nn.Linear(token_hidden * 2, input_dim)
        self.lstm_s = nn.LSTM(input_size=input_dim, **lstm_kw)
        self.lstm_w = nn.LSTM(input_size=input_dim, **lstm_kw)
        self.res_s = nn.Linear(input_dim, h2)
        self.res_w = nn.Linear(input_dim, h2)
        self.head_s = _make_mlp_head(h2, dropout)
        self.head_w = _make_mlp_head(h2, dropout)
        self.head_s_start = _make_mlp_head(h2, dropout)
        self.head_w_start = _make_mlp_head(h2, dropout)
        self.merge_logit = nn.Parameter(torch.zeros(1))
        self.kw_logit_coh = nn.Parameter(torch.zeros(1))
        self.kw_logit_bnd = nn.Parameter(torch.zeros(1))
        # Utterance Boundary Informativeness Weighting (UBIW)
        # A shared scorer over BiLSTM outputs of both streams. Maps h2-dim
        # contextual features → scalar informativeness ∈ (0,1) per utterance.
        # Applied as residual amplification: feat * (1 + strength * w_i).
        # strength is initialised to sigmoid(−3) ≈ 0.05 so the module starts
        # near-neutral and must earn its effect via gradient updates; no
        # external supervision is required.
        if use_ubiw:
            self.ubiw_scorer_end = nn.Sequential(
                nn.Linear(h2, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.ubiw_scorer_start = nn.Sequential(
                nn.Linear(h2, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            )
            self.ubiw_strength = nn.Parameter(torch.tensor(-3.0))
        self.register_buffer("log_pos_weight", torch.zeros((), dtype=torch.float32))
        self.crf = CRF(num_tags=2, batch_first=True) if use_crf else None
        self.topic_keywords = {}
        self.topic_kw_channels = {}
        if os.path.exists(topic_json_path):
            with open(topic_json_path, "r", encoding="utf-8") as f:
                self.topic_keywords = json.load(f)
            self.topic_kw_channels = {
                ds: topic_channel_sets_from_info(info)
                for ds, info in self.topic_keywords.items()
            }

    def set_class_balance(self, pos_weight: float) -> None:
        self.log_pos_weight.fill_(math.log(max(float(pos_weight), 1e-8)))

    def sync_start_heads_from_end(self) -> None:
        self.head_s_start.load_state_dict(self.head_s.state_dict())
        self.head_w_start.load_state_dict(self.head_w.state_dict())

    def load_state_dict(self, state_dict, strict: bool = True):
        d = dict(state_dict)
        # backward compat: old checkpoints used kw_scale_logit / kw_scale_logit_ae
        if "kw_logit_coh" not in d:
            if "kw_scale_logit_ae" in d:
                d["kw_logit_coh"] = d.pop("kw_scale_logit_ae")
            elif "kw_scale_logit" in d:
                d["kw_logit_coh"] = d.pop("kw_scale_logit")
        if "kw_logit_bnd" not in d and "kw_scale_logit_bd" in d:
            d["kw_logit_bnd"] = d.pop("kw_scale_logit_bd")
        return super().load_state_dict(d, strict=strict)

    def _branch_feat(
        self,
        x: torch.Tensor,
        lengths: torch.Tensor,
        lstm: nn.LSTM,
        res: nn.Linear,
    ) -> torch.Tensor:
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = lstm(packed)
        umax = x.size(1)
        out, _ = pad_packed_sequence(out, batch_first=True, total_length=umax)
        feat = out + res(x[:, :umax, :])
        return feat

    def _apply_ubiw(
        self,
        feat: torch.Tensor,
        lengths: torch.Tensor,
        scorer: nn.Module,
    ) -> torch.Tensor:
        if not self.use_ubiw:
            return feat
        bsz, umax, _ = feat.shape
        info_w = torch.sigmoid(scorer(feat))
        time_idx = torch.arange(umax, device=feat.device).unsqueeze(0)
        valid = (time_idx < lengths.to(feat.device).unsqueeze(1)).unsqueeze(-1).to(feat.dtype)
        denom = valid.sum(dim=1, keepdim=True).clamp(min=1.0)
        info_center = (info_w * valid).sum(dim=1, keepdim=True) / denom
        centered_w = (info_w - info_center) * valid
        strength = torch.sigmoid(self.ubiw_strength)
        return feat * (1.0 + strength * centered_w)

    def _tokens_to_utterance_vecs(
        self,
        x_w: torch.Tensor,
        tok_mask: torch.Tensor,
        utt_lengths: torch.Tensor,
    ) -> torch.Tensor:
        B, umax, L, d = x_w.shape
        flat_x = x_w.view(B * umax, L, d)
        flat_m = tok_mask.view(B * umax, L)
        has_tok = flat_m.sum(dim=1) > 0.5
        tl = flat_m.sum(dim=1).long().clamp(min=1)
        for b in range(B):
            n = int(utt_lengths[b].item())
            for j in range(umax):
                idx = b * umax + j
                if j >= n:
                    tl[idx] = 1
        pos = self.pos_emb[:, :L, :].to(flat_x.dtype)
        x_tf = flat_x + pos
        pm = flat_m <= 0.5
        all_pad = pm.all(dim=1, keepdim=True)
        j_idx = torch.arange(L, device=x_w.device).unsqueeze(0)
        pad_mask = pm & (j_idx > 0 | ~all_pad)
        if self.use_token_transformer:
            flat_enc = self.token_transformer(x_tf, src_key_padding_mask=pad_mask)
        else:
            flat_enc = x_tf
        packed = pack_padded_sequence(flat_enc, tl.cpu(), batch_first=True, enforce_sorted=False)
        tok_out, _ = self.token_lstm(packed)
        tok_out, _ = pad_packed_sequence(
            tok_out, batch_first=True, total_length=L
        )
        m_exp = flat_m.unsqueeze(-1)
        pos_u = torch.linspace(0.0, 1.0, steps=L, device=x_w.device, dtype=flat_x.dtype).unsqueeze(0)
        edge_bias = torch.abs(2.0 * pos_u - 1.0).pow(self.edge_gate_gamma)
        pos_gate = self.edge_gate_alpha + (1.0 - self.edge_gate_alpha) * edge_bias
        gate_exp = pos_gate.unsqueeze(-1)
        weighted_mask = m_exp * gate_exp
        denom = weighted_mask.sum(dim=1).clamp(min=1e-6)
        pooled = (tok_out * weighted_mask).sum(dim=1) / denom * has_tok.unsqueeze(-1).float()
        u_flat = self.word_utt_proj(pooled)
        return u_flat.view(B, umax, d)

    def _mix_streams(self, e_s: torch.Tensor, e_w: torch.Tensor) -> torch.Tensor:
        if self.stream_mode == "sentence":
            return e_s
        if self.stream_mode == "token":
            return e_w
        w = torch.sigmoid(self.merge_logit)
        return w * e_s + (1.0 - w) * e_w

    def _apply_bias_terms(
        self,
        emissions: torch.Tensor,
        kw_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if kw_scores is not None:
            kw_scores = kw_scores.to(emissions.device, dtype=emissions.dtype)
            if kw_scores.dim() == 2:
                kw_coh = kw_scores
                kw_bnd = torch.zeros_like(kw_coh)
            else:
                kw_coh = kw_scores[..., 0]
                kw_bnd = kw_scores[..., 1]
            s_coh = torch.sigmoid(self.kw_logit_coh)
            s_bnd = torch.sigmoid(self.kw_logit_bnd)
            boost = s_coh * kw_coh + s_bnd * kw_bnd
            e1 = emissions[:, :, 1] + boost
            emissions = torch.stack([emissions[:, :, 0], e1], dim=-1)
        e1 = emissions[:, :, 1] + self.log_pos_weight
        return torch.stack([emissions[:, :, 0], e1], dim=-1)

    def forward_heads(
        self,
        x_s: torch.Tensor,
        x_w: torch.Tensor,
        tok_mask: torch.Tensor,
        lengths: torch.Tensor,
        kw_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        f_s = self._branch_feat(x_s, lengths, self.lstm_s, self.res_s)
        x_w_u = self._tokens_to_utterance_vecs(x_w, tok_mask, lengths)
        f_w = self._branch_feat(x_w_u, lengths, self.lstm_w, self.res_w)

        if self.use_ubiw:
            f_s_end = self._apply_ubiw(f_s, lengths, self.ubiw_scorer_end)
            f_w_end = self._apply_ubiw(f_w, lengths, self.ubiw_scorer_end)
            f_s_start = self._apply_ubiw(f_s, lengths, self.ubiw_scorer_start)
            f_w_start = self._apply_ubiw(f_w, lengths, self.ubiw_scorer_start)
        else:
            f_s_end, f_w_end = f_s, f_w
            f_s_start, f_w_start = f_s, f_w

        end_s = self.head_s(f_s_end)
        end_w = self.head_w(f_w_end)
        start_s = self.head_s_start(f_s_start)
        start_w = self.head_w_start(f_w_start)

        end_emissions = self._apply_bias_terms(self._mix_streams(end_s, end_w), kw_scores=kw_scores)
        start_emissions = self._apply_bias_terms(self._mix_streams(start_s, start_w), kw_scores=kw_scores)
        cut_emissions = 0.5 * end_emissions + 0.5 * torch.roll(start_emissions, shifts=-1, dims=1)
        lengths = lengths.to(x_s.device)
        time_idx = torch.arange(cut_emissions.size(1), device=x_s.device).unsqueeze(0)
        mask = time_idx < lengths.unsqueeze(1)
        return cut_emissions, end_emissions, start_emissions, mask

    def forward(
        self,
        x_s: torch.Tensor,
        x_w: torch.Tensor,
        tok_mask: torch.Tensor,
        lengths: torch.Tensor,
        kw_scores: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cut_emissions, _, _, mask = self.forward_heads(
            x_s, x_w, tok_mask, lengths, kw_scores=kw_scores
        )
        return cut_emissions, mask

    def get_ubiw_weights_dual(
        self,
        x_s: torch.Tensor,
        lengths: torch.Tensor,
        detach: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if not self.use_ubiw:
            return None

        def _compute() -> tuple[torch.Tensor, torch.Tensor]:
            packed = pack_padded_sequence(
                x_s, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm_s(packed)
            umax = x_s.size(1)
            out, _ = pad_packed_sequence(out, batch_first=True, total_length=umax)
            feat = out + self.res_s(x_s[:, :umax, :])
            info_end = torch.sigmoid(self.ubiw_scorer_end(feat)).squeeze(-1)
            info_start = torch.sigmoid(self.ubiw_scorer_start(feat)).squeeze(-1)
            return info_end, info_start

        if detach:
            with torch.no_grad():
                return _compute()
        return _compute()

    def get_ubiw_weights(
        self,
        x_s: torch.Tensor,
        lengths: torch.Tensor,
        detach: bool = True,
    ) -> torch.Tensor | None:
        """Return per-utterance boundary informativeness weights for post-hoc analysis.

        Args:
            x_s:     Sentence-stream embeddings (B, T, input_dim).
            lengths: Valid utterance counts per sample (B,).

        Returns:
            Tensor of shape (B, T) with values in (0, 1), or None if use_ubiw=False.
            Higher values indicate utterances that the model has learned to treat as
            boundary-informative (e.g. procedural markers in VHF, topic-transition
            phrases in general dialogue).
        """
        if not self.use_ubiw:
            return None
        def _compute() -> torch.Tensor:
            dual = self.get_ubiw_weights_dual(x_s, lengths, detach=False)
            info_end, info_start = dual
            return 0.5 * (info_end + info_start)

        if detach:
            with torch.no_grad():
                return _compute()
        return _compute()
