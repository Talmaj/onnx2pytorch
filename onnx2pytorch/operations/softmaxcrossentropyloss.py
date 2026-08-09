import torch
from torch import nn
from torch.nn import functional as F


class SoftmaxCrossEntropyLoss(nn.Module):
    """ONNX SoftmaxCrossEntropyLoss: cross entropy of log softmax scores and labels."""

    def __init__(self, reduction="mean", ignore_index=None):
        super().__init__()
        self.reduction = reduction
        # Labels are class indices, so the torch default never matches one
        self.ignore_index = -100 if ignore_index is None else ignore_index

    def forward(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        weights: torch.Tensor = None,
    ):
        log_prob = F.log_softmax(scores, dim=1)
        loss = F.nll_loss(
            log_prob,
            labels.to(torch.long),
            weight=weights,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )
        return loss, log_prob
