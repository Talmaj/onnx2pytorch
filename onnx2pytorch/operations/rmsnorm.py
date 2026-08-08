import torch
from torch import nn


class RMSNormalization(nn.Module):
    def __init__(self, dim=-1, eps=1e-5, stash_type=1):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.stash_type = stash_type

    def forward(self, X: torch.Tensor, scale: torch.Tensor):
        axis = self.dim if self.dim >= 0 else X.ndim + self.dim
        dims = tuple(range(axis, X.ndim))
        norm = torch.rsqrt(
            torch.mean(X.float() ** 2, dim=dims, keepdim=True) + self.eps
        )
        return (X.float() * norm).to(X.dtype) * scale

    def extra_repr(self) -> str:
        return "dim={}, eps={}".format(self.dim, self.eps)
