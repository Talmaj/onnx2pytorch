import torch
from torch import nn


class Unique(nn.Module):
    def __init__(self, dim=None, sorted=1):
        super().__init__()
        self.dim = dim
        self.sorted = bool(sorted)

    def forward(self, X: torch.Tensor):
        if self.dim is None:
            data = X.flatten()
            dim = 0
        else:
            data = X
            dim = self.dim if self.dim >= 0 else self.dim + X.ndim

        y, inverse, counts = torch.unique(
            data, sorted=True, return_inverse=True, return_counts=True, dim=dim
        )
        inverse = inverse.flatten()
        num_unique = y.shape[dim]

        positions = torch.arange(inverse.numel(), device=X.device)
        indices = torch.full(
            (num_unique,), inverse.numel(), dtype=torch.long, device=X.device
        )
        indices = indices.scatter_reduce(0, inverse, positions, reduce="amin")

        if not self.sorted:
            # ONNX keeps the order of first occurrence when sorted=0
            order = torch.argsort(indices)
            y = torch.index_select(y, dim, order)
            indices = indices[order]
            counts = counts[order]
            remap = torch.empty_like(order)
            remap[order] = torch.arange(num_unique, device=X.device)
            inverse = remap[inverse]

        return y, indices, inverse, counts.long()

    def extra_repr(self) -> str:
        return "dim={}, sorted={}".format(self.dim, self.sorted)
