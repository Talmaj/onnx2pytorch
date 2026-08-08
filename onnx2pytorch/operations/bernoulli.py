import torch
from torch import nn

from onnx2pytorch.utils import get_random_generator, get_torch_dtype


class Bernoulli(nn.Module):
    """ONNX Bernoulli: draw binary samples from the probabilities in the input."""

    def __init__(self, dtype=None, seed=None):
        super().__init__()
        self.dtype = dtype
        self.seed = seed

    def forward(self, input: torch.Tensor):
        dtype = input.dtype if self.dtype is None else get_torch_dtype(self.dtype)
        generator = get_random_generator(self.seed, input.device)
        output = torch.bernoulli(input.to(torch.float32), generator=generator)
        return output.to(dtype)
