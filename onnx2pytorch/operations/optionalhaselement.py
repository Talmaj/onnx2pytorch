import torch
from torch import nn


class OptionalHasElement(nn.Module):
    """
    Outputs whether the optional-type input contains an element.

    Optionals are represented as the contained value, or None when empty. An
    omitted input counts as empty. A contained empty sequence still counts as
    an element.
    """

    def forward(self, input=None):
        return torch.tensor(input is not None)
