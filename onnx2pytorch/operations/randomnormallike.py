import torch
from torch import nn

from onnx2pytorch.utils import get_random_generator, get_torch_dtype


class RandomNormalLike(nn.Module):
    """ONNX RandomNormalLike: normal samples with the shape of the input tensor."""

    def __init__(self, dtype=None, mean=0.0, scale=1.0, seed=None):
        super().__init__()
        self.dtype = dtype
        self.mean = mean
        self.scale = scale
        self.seed = seed

    def forward(self, input: torch.Tensor):
        dtype = input.dtype if self.dtype is None else get_torch_dtype(self.dtype)
        generator = get_random_generator(self.seed, input.device)
        output = torch.randn(
            input.shape, generator=generator, dtype=torch.float32, device=input.device
        )
        return (output * self.scale + self.mean).to(dtype)
