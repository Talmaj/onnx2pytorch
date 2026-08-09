import torch
from torch import nn


class Dropout(nn.Module):
    """
    ONNX Dropout in inference mode: an identity plus a mask.

    Up to opset 11 the mask marks the dropped elements, so it is all false,
    and before opset 10 it has the same type as the data. From opset 12 on it
    marks the kept elements, so it is all true.
    """

    def __init__(self, opset_version=13, ratio=None, seed=None):
        super().__init__()
        self.opset_version = opset_version
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
        if self.opset_version >= 12:
            mask = torch.ones_like(data, dtype=torch.bool)
        elif self.opset_version >= 10:
            mask = torch.zeros_like(data, dtype=torch.bool)
        else:
            mask = torch.zeros_like(data)
        return data, mask
