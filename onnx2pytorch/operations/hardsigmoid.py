import math

import torch
from torch import nn


class Hardsigmoid(nn.Module):
    def __new__(cls, alpha=0.2, beta=0.5):
        """
        If alpha and beta same as default values for torch's Hardsigmoid,
        return torch's Hardsigmoid. Else, return custom Hardsigmoid.
        """
        if math.isclose(alpha, 1 / 6, abs_tol=1e-2) and beta == 0.5:
            return nn.Hardsigmoid()
        else:
            return super().__new__(cls)

    def __init__(self, alpha=0.2, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input):
        return torch.clip(input * self.alpha + self.beta, 0, 1)
