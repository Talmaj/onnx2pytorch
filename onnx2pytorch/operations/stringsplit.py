import numpy as np
import torch
from torch import nn


class StringSplit(nn.Module):
    """
    ONNX StringSplit: split each element of a string tensor on a delimiter.

    The substrings are padded with empty strings to a rectangular tensor with one
    extra trailing dimension. The second output holds the number of substrings
    each element was split into, which is zero for empty input strings.
    """

    def __init__(self, delimiter=None, maxsplit=None):
        super().__init__()
        # An empty delimiter means splitting on consecutive whitespace
        self.delimiter = delimiter or None
        self.maxsplit = maxsplit

    def forward(self, X: np.ndarray):
        x = np.asarray(X).astype(np.str_)
        maxsplit = -1 if self.maxsplit is None else self.maxsplit
        substrings = [
            [] if s == "" else s.split(self.delimiter, maxsplit)
            for s in x.reshape(-1).tolist()
        ]
        num_substrings = [len(parts) for parts in substrings]
        width = max(num_substrings, default=0)

        splits = np.array(
            [parts + [""] * (width - len(parts)) for parts in substrings], dtype=object
        ).reshape(*x.shape, width)
        num_splits = torch.tensor(num_substrings, dtype=torch.int64).reshape(
            tuple(x.shape)
        )
        return splits, num_splits

    def extra_repr(self) -> str:
        return "delimiter={!r}, maxsplit={}".format(self.delimiter, self.maxsplit)
