from functools import partial

import torch
from torch import nn


class Concat(nn.Module):
    def __init__(self, **attributes):
        self.ordered_attrs = []
        for name, val in attributes.items():
            setattr(self, name, val)
            self.ordered_attrs.append(name)
        super().__init__()

    def forward(self, *inputs):
        attributes = {
            name: getattr(self, name).to(inputs[0].device)
            for name in self.ordered_attrs
        }
        op = partial(torch.cat, **attributes)
        return op(inputs)
