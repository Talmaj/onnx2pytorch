import torch
from torch import nn
from torch.nn import functional as F


class NegativeLogLikelihoodLoss(nn.Module):
    """ONNX NegativeLogLikelihoodLoss: negative log likelihood of the target classes."""

    def __init__(self, reduction="mean", ignore_index=None):
        super().__init__()
        self.reduction = reduction
        # Targets are class indices, so the torch default never matches one
        self.ignore_index = -100 if ignore_index is None else ignore_index

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor = None,
    ):
        return F.nll_loss(
            input,
            target.to(torch.long),
            weight=weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
