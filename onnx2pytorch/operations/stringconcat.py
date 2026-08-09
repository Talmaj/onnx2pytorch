import numpy as np
from torch import nn


class StringConcat(nn.Module):
    """ONNX StringConcat: elementwise concatenation of two string tensors."""

    def forward(self, X: np.ndarray, Y: np.ndarray):
        x = np.asarray(X).astype(np.str_)
        y = np.asarray(Y).astype(np.str_)
        return np.char.add(x, y).astype(object)
