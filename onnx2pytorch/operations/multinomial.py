import torch
from torch import nn
from torch.nn import functional as F

from onnx2pytorch.utils import get_random_generator, get_torch_dtype


class Multinomial(nn.Module):
    """ONNX Multinomial: sample from the unnormalized log probabilities in the input."""

    def __init__(self, dtype=6, sample_size=1, seed=None):
        super().__init__()
        self.dtype = dtype
        self.sample_size = sample_size
        self.seed = seed

    def forward(self, input: torch.Tensor):
        dtype = get_torch_dtype(self.dtype)
        generator = get_random_generator(self.seed, input.device)
        probabilities = F.softmax(input.to(torch.float32), dim=1)
        output = torch.multinomial(
            probabilities, self.sample_size, replacement=True, generator=generator
        )
        return output.to(dtype)
