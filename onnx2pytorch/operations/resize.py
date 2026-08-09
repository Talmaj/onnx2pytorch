import warnings

import torch
from torch.nn import functional as F

from onnx2pytorch.operations.base import Operator

empty_tensor = torch.Tensor([])

LINEAR_MODES = {1: "linear", 2: "bilinear", 3: "trilinear"}


def linear_interpolate_asymmetric(inp, scales):
    """
    Linear interpolation with the coordinate transform used before opset 11.

    Upsample and Resize-10 map an output index to input index / scale, while
    torch's interpolate only offers the half_pixel and align_corners variants.
    """
    out = inp
    for axis, scale in enumerate(scales):
        dim = axis + 2
        in_size = inp.shape[dim]
        src = torch.arange(int(in_size * scale), device=inp.device) / scale
        src = src.clamp(0, in_size - 1)
        low = src.floor().long()
        high = torch.clamp(low + 1, max=in_size - 1)
        shape = [1] * out.dim()
        shape[dim] = src.numel()
        weight = (src - low).to(out.dtype).reshape(shape)
        out = (
            out.index_select(dim, low) * (1 - weight)
            + out.index_select(dim, high) * weight
        )
    return out


class Resize(Operator):
    def __init__(self, opset_version=13, mode="nearest", align_corners=None, **kwargs):
        self.opset_version = opset_version
        self.mode = mode
        self.align_corners = align_corners
        for key in kwargs.keys():
            warnings.warn(
                "Pytorch's interpolate uses no {}. " "Result might differ.".format(key)
            )
        super().__init__()

    def torch_mode(self, spatial_dims):
        """Pytorch names the linear and cubic modes after the input rank."""
        if self.mode == "linear":
            if spatial_dims not in LINEAR_MODES:
                raise NotImplementedError(
                    "Pytorch's interpolate has no linear mode for {} spatial "
                    "dimensions.".format(spatial_dims)
                )
            return LINEAR_MODES[spatial_dims]
        elif self.mode == "cubic":
            if spatial_dims != 2:
                raise NotImplementedError(
                    "Pytorch's interpolate has no cubic mode for {} spatial "
                    "dimensions.".format(spatial_dims)
                )
            return "bicubic"
        return self.mode

    def interpolate(self, inp, roi=None, scales=None, sizes=None):
        # Optional inputs that were omitted in the onnx graph are passed on as None
        roi = empty_tensor if roi is None else roi
        scales = empty_tensor if scales is None else scales
        sizes = empty_tensor if sizes is None else sizes

        if roi.nelement() > 0:
            warnings.warn("Pytorch's interpolate uses no roi. Result might differ.")

        # Interpolate does not accept the tensors that the onnx graph provides
        scales = [float(scale) for scale in scales]
        sizes = [int(size) for size in sizes]
        shape = list(inp.shape)
        if shape[:2] == sizes[:2]:
            sizes = sizes[2:]  # Pytorch's interpolate takes only H and W params
        elif scales[:2] == [1, 1]:
            scales = scales[2:]
        elif len(scales) == 0 and len(sizes) == 0:
            raise ValueError("One of the two, scales or sizes, needs to be defined.")
        else:
            raise NotImplementedError(
                "Pytorch's interpolate does not scale batch and channel dimensions."
            )

        if self.opset_version < 11 and self.mode == "linear" and scales:
            return linear_interpolate_asymmetric(inp, scales)

        if len(scales) == 0:
            scales = None
        elif len(sizes) == 0:
            sizes = None
        else:
            raise ValueError(
                "Only one of the two, scales or sizes, needs to be defined."
            )

        return F.interpolate(
            inp,
            scale_factor=scales,
            size=sizes,
            mode=self.torch_mode(inp.dim() - 2),
            align_corners=self.align_corners,
        )

    def forward(self, inp, roi=None, scales=None, sizes=None):
        if self.opset_version < 11:
            # Resize-10 has the signature (X, scales)
            roi, scales = None, roi
        return self.interpolate(inp, roi, scales, sizes)


class Upsample(Resize):
    """Deprecated onnx operator."""

    def __init__(
        self,
        opset_version=9,
        mode="nearest",
        scales=None,
        height_scale=None,
        width_scale=None,
        **kwargs,
    ):
        super().__init__(opset_version=opset_version, mode=mode, **kwargs)
        if scales is not None:
            self.scales = list(scales)
        elif height_scale is not None or width_scale is not None:
            # Upsample-1 names the two spatial scales individually
            self.scales = [1.0, 1.0, height_scale or 1.0, width_scale or 1.0]
        else:
            self.scales = None

    def forward(self, inp, scales=None):
        if scales is None:
            if self.scales is None:
                raise TypeError(
                    "forward() missing 1 required positional argument: 'scales'"
                )
            scales = self.scales
        return self.interpolate(inp, None, scales, None)
