import torch
from torch import nn

from onnx2pytorch.utils import get_random_generator, get_torch_dtype


class RandomUniformLike(nn.Module):
    """
    ONNX RandomUniformLike: uniform samples with the shape of the input tensor.

    Parameters
    ----------
    dtype : int, optional
        The data type for the elements of the output tensor (ONNX TensorProto type).
        If not specified, uses the data type of the input tensor.
    high : float, default=1.0
        Upper boundary of the output values.
    low : float, default=0.0
        Lower boundary of the output values.
    seed : float, optional
        Seed to the random generator. If not specified, will auto generate one.
    """

    def __init__(self, dtype=None, high=1.0, low=0.0, seed=None):
        super().__init__()
        self.dtype = dtype
        self.high = high
        self.low = low
        self.seed = seed

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        dtype = input.dtype if self.dtype is None else get_torch_dtype(self.dtype)
        generator = get_random_generator(self.seed, input.device)
        # torch.rand needs a floating point dtype, the cast comes after scaling
        output = torch.rand(
            input.shape, generator=generator, dtype=torch.float32, device=input.device
        )
        return (output * (self.high - self.low) + self.low).to(dtype)
