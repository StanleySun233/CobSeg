import torch
import torch.nn as nn
"""
Sentence-only DTS model definition.
Only model structure is provided here (no training / CLI / data pipeline).
"""


def _make_mlp_head(hidden_in: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_in, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 2),
    )


class PureBertSegmenter(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        dropout = 0.5
        self.head = _make_mlp_head(input_dim, dropout)
        self.register_buffer("ce_weight", torch.tensor([1.0, 1.0], dtype=torch.float32))

    def forward(
        self,
        x_s: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.head(x_s)
        lengths = lengths.to(x_s.device)
        time_idx = torch.arange(logits.size(1), device=x_s.device).unsqueeze(0)
        mask = time_idx < lengths.unsqueeze(1)
        return logits, mask
