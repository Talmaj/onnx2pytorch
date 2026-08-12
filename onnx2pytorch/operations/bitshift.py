import torch
from torch import nn


class BitShift(nn.Module):
    def __init__(self, direction):
        if direction not in ("LEFT", "RIGHT"):
            raise ValueError("invalid BitShift direction {}".format(direction))

        self.direction = direction
        super().__init__()

    def forward(self, X, Y):
        # Shifting is only implemented for uint8 among the unsigned types, so the
        # ones that fit in an int64 are shifted there and cast back. uint64 has
        # no room for that and keeps failing loudly.
        dtype = X.dtype
        if dtype in (torch.uint16, torch.uint32):
            X, Y = X.long(), Y.long()
        if self.direction == "LEFT":
            shifted = X << Y
        else:
            shifted = X >> Y
        return shifted.to(dtype)
