from torch import nn


class Optional(nn.Module):
    """
    Optional operation.

    Constructs an optional-type value containing either an empty optional
    or a non-empty value containing the input element.

    In PyTorch, optionals are represented as:
    - None for empty optional
    - The actual value for non-empty optional
    """

    def __init__(self, type=None):
        """
        Parameters
        ----------
        type : optional
            ONNX type specification for empty optional. Not used in PyTorch.
        """
        super().__init__()
        self.type = type

    def forward(self, input=None):
        """
        Create an optional value.

        Parameters
        ----------
        input : torch.Tensor, list, or None
            The input element. If None or not provided, creates an empty optional.
            Can be a tensor, sequence (list), or any other value.

        Returns
        -------
        value or None
            The input value if provided, None for empty optional.
        """
        # If no input provided, return None (empty optional)
        if input is None:
            return None

        # Return the input value (non-empty optional)
        return input
