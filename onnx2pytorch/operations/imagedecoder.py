import torch
from torch import nn
from torchvision.io import decode_image, ImageReadMode


class ImageDecoder(nn.Module):
    """ONNX ImageDecoder: decode an encoded image into an uint8 HWC tensor."""

    def __init__(self, pixel_format="RGB"):
        super().__init__()
        if pixel_format not in ("RGB", "BGR", "Grayscale"):
            raise NotImplementedError(
                "ImageDecoder with pixel_format={} not implemented.".format(
                    pixel_format
                )
            )
        self.pixel_format = pixel_format

    def forward(self, encoded: torch.Tensor):
        mode = (
            ImageReadMode.GRAY
            if self.pixel_format == "Grayscale"
            else ImageReadMode.RGB
        )
        decoded = decode_image(encoded.to(torch.uint8).cpu(), mode=mode)
        decoded = decoded.permute(1, 2, 0)
        if self.pixel_format == "BGR":
            decoded = decoded.flip(-1)
        return decoded.contiguous()
