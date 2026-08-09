import numpy as np
import torch
from torch import nn


class ConstantOfShape(nn.Module):
    def __init__(self, constant=None):
        super().__init__()
        if constant is None:
            # ONNX defaults to a float32 zero when the value attribute is absent
            const = torch.tensor(0.0, dtype=torch.float32)
        else:
            const = torch.tensor(np.copy(constant))
        self.register_buffer("constant", const)

    def forward(self, shape: torch.Tensor):
        return self.constant.expand(*shape).to(shape.device)

    def extra_repr(self) -> str:
        return "constant={}".format(self.constant)
