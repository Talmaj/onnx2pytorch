import torch
from torch import nn


class Div(nn.Module):
    """ONNX Div, which truncates towards zero for integers rather than flooring."""

    def forward(self, input, other):
        res_type = torch.result_type(input, other)
        true_quotient = torch.true_divide(input, other)
        if res_type.is_floating_point:
            return true_quotient
        return torch.trunc(true_quotient).to(res_type)
