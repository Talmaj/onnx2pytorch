import torch
from torch import nn
from torchvision.ops import roi_pool


class MaxRoiPool(nn.Module):
    def __init__(self, pooled_shape, spatial_scale=1.0):
        super().__init__()
        self.pooled_shape = tuple(pooled_shape)
        self.spatial_scale = spatial_scale

    def forward(self, X: torch.Tensor, rois: torch.Tensor):
        return roi_pool(
            X,
            rois.to(X.dtype),
            output_size=self.pooled_shape,
            spatial_scale=self.spatial_scale,
        )

    def extra_repr(self) -> str:
        return "pooled_shape={}, spatial_scale={}".format(
            self.pooled_shape, self.spatial_scale
        )
