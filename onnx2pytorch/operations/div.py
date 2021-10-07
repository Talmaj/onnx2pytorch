import torch
from torch import nn


class Div(nn.Module):
    def forward(self, input, other):
        res_type = torch.result_type(input, other)
        true_quotient = torch.true_divide(input, other)
        if res_type.is_floating_point:
            res = true_quotient
        else:
            res = torch.floor(true_quotient).to(res_type)
        return res
