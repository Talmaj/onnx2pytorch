import torch
from torch import nn

from onnx2pytorch.utils import get_random_generator, get_torch_dtype


class RandomNormal(nn.Module):
    """ONNX RandomNormal: tensor of the given shape drawn from a normal distribution."""

    def __init__(self, shape, dtype=1, mean=0.0, scale=1.0, seed=None):
        super().__init__()
        self.shape = shape
        self.dtype = dtype
        self.mean = mean
        self.scale = scale
        self.seed = seed

    def forward(self):
        dtype = get_torch_dtype(self.dtype)
        generator = get_random_generator(self.seed)
        output = torch.randn(tuple(self.shape), generator=generator)
        return (output * self.scale + self.mean).to(dtype)
