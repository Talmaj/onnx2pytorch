from torch import nn


class SequenceEmpty(nn.Module):
    def __init__(self, dtype=None):
        super().__init__()
        self.dtype = dtype

    def forward(self):
        return []

    def extra_repr(self) -> str:
        return "dtype={}".format(self.dtype)
