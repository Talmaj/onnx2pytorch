import torch
from torch import nn


class BitShift(nn.Module):
    def __init__(self, direction):
        if direction not in ("LEFT", "RIGHT"):
            raise ValueError("invalid BitShift direction {}".format(direction))

        self.direction = direction
        super().__init__()

    def forward(self, X, Y):
        if self.direction == "LEFT":
            return X << Y
        elif self.direction == "RIGHT":
            return X >> Y
