import torch
from torch import nn


class Dropout(nn.Module):
    """ONNX Dropout in inference mode: an identity plus an all-true mask."""

    def __init__(self, ratio=None, seed=None):
        super().__init__()
        self.ratio = ratio
        self.seed = seed

    def forward(
        self,
        data: torch.Tensor,
        ratio: torch.Tensor = None,
        training_mode: torch.Tensor = None,
    ):
        if training_mode is not None and bool(training_mode):
            raise NotImplementedError(
                "Dropout with training_mode=True not implemented."
            )
        return data, torch.ones_like(data, dtype=torch.bool)
