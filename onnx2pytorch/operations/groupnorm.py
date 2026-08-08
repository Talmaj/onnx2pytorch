import torch
from torch import nn
from torch.nn import functional as F


class GroupNormalization(nn.Module):
    """ONNX GroupNormalization.

    Up to opset 20 scale and bias are given per group, from opset 21 on
    they are given per channel.
    """

    def __init__(self, num_groups, eps=1e-5, stash_type=1, opset_version=21):
        super().__init__()
        self.num_groups = num_groups
        self.eps = eps
        self.stash_type = stash_type
        self.opset_version = opset_version

    def forward(self, X: torch.Tensor, scale: torch.Tensor, bias: torch.Tensor):
        channels = X.shape[1]
        if self.opset_version < 21 and scale.numel() == self.num_groups:
            repeats = channels // self.num_groups
            scale = scale.repeat_interleave(repeats)
            bias = bias.repeat_interleave(repeats)
        return F.group_norm(X, self.num_groups, scale, bias, self.eps)

    def extra_repr(self) -> str:
        return "num_groups={}, eps={}".format(self.num_groups, self.eps)
