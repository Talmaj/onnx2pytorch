import re

import numpy as np
import torch
from torch import nn


class RegexFullMatch(nn.Module):
    """
    ONNX RegexFullMatch: elementwise full match of a string tensor against a pattern.

    The spec mandates RE2 syntax, which is approximated with Python's re module.
    re.ASCII reproduces RE2's ASCII-only character classes, but patterns relying on
    constructs outside the common subset (e.g. backreferences) may still differ.
    """

    def __init__(self, pattern=""):
        super().__init__()
        self.pattern = pattern
        try:
            self.regex = re.compile(pattern, re.ASCII)
        except re.error as e:
            raise ValueError("Invalid regex pattern {!r}".format(pattern)) from e

    def forward(self, X: np.ndarray):
        x = np.asarray(X).astype(np.str_)
        matches = [self.regex.fullmatch(s) is not None for s in x.reshape(-1).tolist()]
        return torch.tensor(matches, dtype=torch.bool).reshape(tuple(x.shape))

    def extra_repr(self) -> str:
        return "pattern={!r}".format(self.pattern)
