from torch import nn


class OptionalGetElement(nn.Module):
    """
    Outputs the element of an optional-type input.

    Optionals are represented as the contained value, or None when empty.
    """

    def forward(self, input=None):
        if input is None:
            raise ValueError("OptionalGetElement input must not be an empty optional.")
        return input
