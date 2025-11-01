import torch
from torch import nn


class LRN(nn.Module):
    """
    Local Response Normalization operation.

    Applies local response normalization across channels.

    Parameters
    ----------
    alpha : float
        Scaling parameter (default: 0.0001)
    beta : float
        Exponent (default: 0.75)
    bias : float
        Bias parameter (default: 1.0)
    size : int
        Number of channels to sum over
    """

    def __init__(self, alpha=0.0001, beta=0.75, bias=1.0, size=None):
        super().__init__()
        if size is None:
            raise ValueError("size parameter is required for LRN operation")

        self.alpha = alpha
        self.beta = beta
        self.bias = bias
        self.size = size

        # PyTorch's LocalResponseNorm uses the same formula as ONNX LRN
        self.lrn = nn.LocalResponseNorm(size=size, alpha=alpha, beta=beta, k=bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Apply local response normalization.

        Parameters
        ----------
        X : torch.Tensor
            Input tensor of shape (N, C, *)

        Returns
        -------
        torch.Tensor
            Normalized tensor of same shape as input
        """
        return self.lrn(X)
