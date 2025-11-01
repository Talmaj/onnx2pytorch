import torch
from torch import nn

from onnx2pytorch.dtypes import ONNX_DTYPE_TO_TORCH


class RandomUniformLike(nn.Module):
    """
    RandomUniformLike operation.

    Generate a tensor with random values drawn from a uniform distribution.
    The shape of the output tensor is copied from the shape of the input tensor.

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

        # Set seed if provided
        if seed is not None:
            torch.manual_seed(int(seed))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Generate random tensor with same shape as input.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to copy shape and optionally type from.

        Returns
        -------
        torch.Tensor
            Tensor of random values drawn from uniform distribution.
        """
        # Determine output dtype
        if self.dtype is not None:
            # Map ONNX dtype to PyTorch dtype
            output_dtype = ONNX_DTYPE_TO_TORCH.get(self.dtype)
            if output_dtype is None:
                raise ValueError(
                    f"ONNX dtype {self.dtype} is not supported in PyTorch. "
                    f"Supported dtypes: {[k for k, v in ONNX_DTYPE_TO_TORCH.items() if v is not None]}"
                )
        else:
            # Use input dtype
            output_dtype = input.dtype

        # Generate random tensor with uniform distribution in [0, 1)
        # torch.rand only works with floating point dtypes, so generate as float first
        output = torch.rand(input.shape, dtype=torch.float32, device=input.device)

        # Scale to [low, high] range
        output = output * (self.high - self.low) + self.low

        # Convert to target dtype
        output = output.to(output_dtype)

        return output
