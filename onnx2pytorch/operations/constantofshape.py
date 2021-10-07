import numpy as np
import torch
from torch import nn


class ConstantOfShape(nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        if constant is None:
            const = torch.tensor(1.0, dtype=torch.float32)
        else:
            const = torch.from_numpy(np.copy(constant))
        self.register_buffer("constant", const)

    def forward(self, shape: torch.Tensor):
        return self.constant.expand(*shape).to(shape.device)

    def extra_repr(self) -> str:
        return "constant={}".format(self.constant)
