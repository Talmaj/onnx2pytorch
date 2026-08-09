import torch
from torch import nn

from onnx2pytorch.utils import get_torch_dtype


class MelWeightMatrix(nn.Module):
    """ONNX MelWeightMatrix: matrix projecting spectrogram bins onto mel bands."""

    def __init__(self, output_datatype=1):
        super().__init__()
        self.output_datatype = output_datatype

    def forward(
        self,
        num_mel_bins: torch.Tensor,
        dft_length: torch.Tensor,
        sample_rate: torch.Tensor,
        lower_edge_hertz: torch.Tensor,
        upper_edge_hertz: torch.Tensor,
    ):
        num_mel_bins = int(num_mel_bins)
        dft_length = int(dft_length)
        num_spectrogram_bins = dft_length // 2 + 1

        low_mel = 2595 * torch.log10(1 + lower_edge_hertz.to(torch.float64) / 700)
        high_mel = 2595 * torch.log10(1 + upper_edge_hertz.to(torch.float64) / 700)
        mel_step = (high_mel - low_mel) / (num_mel_bins + 2)

        bins = torch.arange(num_mel_bins + 2, dtype=torch.float64)
        bins = 700 * (torch.pow(10, (bins * mel_step + low_mel) / 2595) - 1)
        bins = torch.div(
            (dft_length + 1) * bins, int(sample_rate), rounding_mode="floor"
        )

        low, center, high = bins[:-2], bins[1:-1], bins[2:]
        j = torch.arange(num_spectrogram_bins, dtype=torch.float64).unsqueeze(-1)

        rising = (center - low).clamp(min=1)
        falling = (high - center).clamp(min=1)
        output = torch.zeros(num_spectrogram_bins, num_mel_bins, dtype=torch.float64)
        output = torch.where(
            (j >= low) & (j <= center) & (center > low), (j - low) / rising, output
        )
        output = torch.where((j == center) & (center == low), 1.0, output)
        output = torch.where(
            (j >= center) & (j < high) & (high > center), (high - j) / falling, output
        )
        return output.to(get_torch_dtype(self.output_datatype))
