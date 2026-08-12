import torch
from torch import nn


class TopK(nn.Module):
    def __init__(self, axis=-1, largest=1, sorted=1, k=None):
        self.axis = axis
        self.largest = bool(largest)
        self.sorted = bool(sorted)
        self.k = k
        super().__init__()

    def forward(self, X: torch.Tensor, K: torch.Tensor = None):
        # K is an attribute at opset 1 and an input from opset 10 on
        k = self.k if K is None else int(K)
        if k is None:
            raise TypeError("forward() missing 1 required positional argument: 'K'")
        # A stable sort rather than torch.topk, because onnx breaks ties on the
        # lower index and torch.topk gives no such guarantee.
        dim = self.axis % X.dim()
        values, indices = torch.sort(X, dim=dim, descending=self.largest, stable=True)
        return values.narrow(dim, 0, k), indices.narrow(dim, 0, k)

    def extra_repr(self) -> str:
        return "axis={}, largest={}, sorted={}".format(
            self.axis, self.largest, self.sorted
        )
