from torch import nn


class SequenceConstruct(nn.Module):
    """
    SequenceConstruct operation.

    Construct a tensor sequence containing 'inputs' tensors.
    All tensors in 'inputs' must have the same data type.

    In PyTorch, sequences are represented as Python lists of tensors.
    """

    def __init__(self):
        super().__init__()

    def forward(self, *inputs):
        """
        Construct a sequence from input tensors.

        Parameters
        ----------
        inputs : variable number of torch.Tensor
            Tensors to construct sequence from (1 or more).

        Returns
        -------
        list
            List of tensors representing the sequence.
        """
        if len(inputs) == 0:
            raise ValueError("SequenceConstruct requires at least one input tensor")

        # Validate all inputs have the same dtype
        if len(inputs) > 1:
            first_dtype = inputs[0].dtype
            for i, tensor in enumerate(inputs[1:], 1):
                if tensor.dtype != first_dtype:
                    raise TypeError(
                        f"All input tensors must have the same dtype. "
                        f"Got {first_dtype} at index 0 and {tensor.dtype} at index {i}"
                    )

        # Return as a list (representing a sequence)
        return list(inputs)
