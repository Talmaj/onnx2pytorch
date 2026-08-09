import torch
from torch import nn

TORCH_REDUCTIONS = {"add": "sum", "mul": "prod", "max": "amax", "min": "amin"}


class ScatterElements(nn.Module):
    def __init__(self, dim=0, reduction="none"):
        self.dim = dim
        self.reduction = reduction
        super().__init__()

    def forward(self, data: torch.Tensor, indices: torch.Tensor, updates: torch.Tensor):
        indices = torch.where(indices < 0, indices + data.size(self.dim), indices)
        if self.reduction == "none":
            return torch.scatter(data, self.dim, indices, updates)
        return torch.scatter_reduce(
            data,
            self.dim,
            indices,
            updates,
            reduce=TORCH_REDUCTIONS[self.reduction],
            include_self=True,
        )

    def extra_repr(self) -> str:
        return "dim={}, reduction={}".format(self.dim, self.reduction)
