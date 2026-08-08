import torch
from torch import nn
from torchvision.ops import roi_align


class RoiAlign(nn.Module):
    def __init__(
        self,
        output_height=1,
        output_width=1,
        sampling_ratio=0,
        spatial_scale=1.0,
        mode="avg",
        coordinate_transformation_mode=None,
        opset_version=16,
    ):
        super().__init__()
        if mode != "avg":
            raise NotImplementedError("RoiAlign mode={} not implemented.".format(mode))
        if coordinate_transformation_mode is None:
            coordinate_transformation_mode = (
                "half_pixel" if opset_version >= 16 else "output_half_pixel"
            )
        self.output_size = (output_height, output_width)
        self.sampling_ratio = sampling_ratio
        self.spatial_scale = spatial_scale
        self.aligned = coordinate_transformation_mode == "half_pixel"

    def forward(self, X: torch.Tensor, rois: torch.Tensor, batch_indices: torch.Tensor):
        boxes = torch.cat([batch_indices.to(rois.dtype).unsqueeze(1), rois], dim=1)
        return roi_align(
            X,
            boxes,
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
            sampling_ratio=self.sampling_ratio if self.sampling_ratio > 0 else -1,
            aligned=self.aligned,
        )

    def extra_repr(self) -> str:
        return "output_size={}, spatial_scale={}, sampling_ratio={}".format(
            self.output_size, self.spatial_scale, self.sampling_ratio
        )
