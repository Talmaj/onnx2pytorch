import warnings

from torch.nn import functional as F

from onnx2pytorch.operations.base import Operator


class Resize(Operator):
    def __init__(self, mode="nearest", align_corners=None, **kwargs):
        self.mode = mode
        self.align_corners = align_corners
        for key in kwargs.keys():
            warnings.warn(
                "Pytorch's interpolate uses no {}. " "Result might differ.".format(key)
            )
        super().__init__()

    def forward(self, inp, roi, scales, sizes):
        if roi.nelement() > 0:
            warnings.warn("Pytorch's interpolate uses no roi. " "Result might differ.")

        scales = scales.tolist()
        sizes = sizes.tolist()
        shape = list(inp.shape)
        if shape[:2] == sizes[:2]:
            sizes = sizes[2:]  # Pytorch's interpolate takes only H and W params
        elif shape[:2] == sizes[:2]:
            sizes = sizes[2:]
        elif len(scales) == 0 and len(sizes) == 0:
            raise ValueError("One of the two, scales or sizes, needs to be defined.")
        else:
            raise NotImplementedError(
                "Pytorch's interpolate does not scale batch and channel dimensions."
            )

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
            mode=self.mode,
            align_corners=self.align_corners,
        )
