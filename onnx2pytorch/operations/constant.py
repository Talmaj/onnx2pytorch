import numpy as np
import torch
from torch import nn


class ConstantOfShape(nn.Module):
    def __init__(self, constant):
        super().__init__()
        self.constant = torch.from_numpy(np.copy(constant))

    def forward(self, shape: torch.Tensor):
        return self.constant * torch.ones(*shape)

    def extra_repr(self) -> str:
        return "constant={}".format(self.constant)
