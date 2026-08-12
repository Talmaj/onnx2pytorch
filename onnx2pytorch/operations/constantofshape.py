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
            # The value attribute is a one element tensor, the fill is its scalar
            const = torch.tensor(np.copy(constant)).reshape(-1)[0]
        self.register_buffer("constant", const)

    def forward(self, shape: torch.Tensor):
        # A shape tensor of length 0 asks for a scalar, which expand spells as an
        # empty size rather than as no argument at all.
        return self.constant.expand(torch.Size(shape)).to(shape.device)

    def extra_repr(self) -> str:
        return "constant={}".format(self.constant)
