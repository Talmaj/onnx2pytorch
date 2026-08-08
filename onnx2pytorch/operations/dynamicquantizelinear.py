import torch
from torch import nn


class DynamicQuantizeLinear(nn.Module):
    """ONNX DynamicQuantizeLinear: quantize to uint8 with a scale derived from x."""

    def forward(self, x: torch.Tensor):
        qmin, qmax = 0.0, 255.0
        x = x.to(torch.float32)
        x_min = torch.clamp(x.min(), max=0.0)
        x_max = torch.clamp(x.max(), min=0.0)

        y_scale = (x_max - x_min) / (qmax - qmin)
        y_zero_point = torch.round(torch.clamp(qmin - x_min / y_scale, qmin, qmax))
        y = torch.clamp(torch.round(x / y_scale) + y_zero_point, qmin, qmax)
        return y.to(torch.uint8), y_scale, y_zero_point.to(torch.uint8)
