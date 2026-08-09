import torch
from torch import nn

from onnx2pytorch.utils import get_random_generator, get_torch_dtype


class RandomUniform(nn.Module):
    """ONNX RandomUniform: tensor of the given shape drawn from a uniform distribution."""

    def __init__(self, shape, dtype=1, high=1.0, low=0.0, seed=None):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.high = high
        self.low = low
        self.seed = seed

    def forward(self):
        dtype = get_torch_dtype(self.dtype)
        generator = get_random_generator(self.seed)
        output = torch.rand(tuple(self.shape), generator=generator)
        return (output * (self.high - self.low) + self.low).to(dtype)
