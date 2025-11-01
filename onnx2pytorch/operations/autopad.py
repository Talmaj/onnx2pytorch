import torch
from torch import nn


class AutoPad(nn.Module):
    """
    Wrapper for automatic padding that computes padding dynamically based on input shape.
    Implements ONNX auto_pad modes: SAME_UPPER, SAME_LOWER, VALID.
    """

    def __init__(self, kernel_size, stride, dilation, mode="SAME_UPPER"):
        super().__init__()
        self.kernel_size = (
            kernel_size
            if isinstance(kernel_size, (tuple, list))
            else (kernel_size,) * 2
        )
        self.stride = stride if isinstance(stride, (tuple, list)) else (stride,) * 2
        self.dilation = (
            dilation if isinstance(dilation, (tuple, list)) else (dilation,) * 2
        )
        self.mode = mode

        if mode not in ("SAME_UPPER", "SAME_LOWER", "VALID"):
            raise ValueError(f"Unsupported auto_pad mode: {mode}")

    def forward(self, x):
        """Compute padding based on input shape."""
        if self.mode == "VALID":
            # No padding
            return x

        # Get spatial dimensions (H, W for 2D, H for 1D, etc.)
        spatial_dims = x.shape[2:]
        ndim = len(spatial_dims)

        # Compute padding for each dimension
        pads = []
        for i in range(ndim):
            input_size = spatial_dims[i]
            kernel_size = (
                self.kernel_size[i]
                if i < len(self.kernel_size)
                else self.kernel_size[-1]
            )
            stride = self.stride[i] if i < len(self.stride) else self.stride[-1]
            dilation = self.dilation[i] if i < len(self.dilation) else self.dilation[-1]

            # Calculate effective kernel size with dilation
            effective_kernel = (kernel_size - 1) * dilation + 1

            # Output size should match input size (for stride=1) or be ceil(input/stride)
            output_size = (input_size + stride - 1) // stride

            # Total padding needed
            total_pad = max(
                0, (output_size - 1) * stride + effective_kernel - input_size
            )

            if self.mode == "SAME_UPPER":
                # More padding on the right/bottom
                pad_before = total_pad // 2
                pad_after = total_pad - pad_before
            else:  # SAME_LOWER
                # More padding on the left/top
                pad_after = total_pad // 2
                pad_before = total_pad - pad_after

            pads.extend([pad_before, pad_after])

        # PyTorch padding order is reversed (last dimension first)
        # For 2D: [left, right, top, bottom]
        pads_reversed = []
        for i in range(ndim - 1, -1, -1):
            pads_reversed.extend([pads[i * 2], pads[i * 2 + 1]])

        # Apply padding
        if any(p > 0 for p in pads_reversed):
            x = torch.nn.functional.pad(x, pads_reversed, mode="constant", value=0)

        return x
