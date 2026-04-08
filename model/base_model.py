import torch
import torch.nn as nn


class BaseModel(nn.Module):
    default_lr = 1e-3
    default_lr_patience = 5
    default_lr_factor = 0.5
    default_min_lr = 1e-6
    default_early_stop = 10

    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def to_device(self):
        return self.to(self.device)
