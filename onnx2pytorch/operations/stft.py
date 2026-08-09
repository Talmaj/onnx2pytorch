import torch
from torch import nn


class STFT(nn.Module):
    """ONNX STFT: short-time Fourier transform of a real or complex signal."""

    def __init__(self, onesided=1):
        super().__init__()
        self.onesided = onesided

    def forward(
        self,
        signal: torch.Tensor,
        frame_step: torch.Tensor,
        window: torch.Tensor = None,
        frame_length: torch.Tensor = None,
    ):
        step = int(frame_step)
        if frame_length is not None:
            length = int(frame_length)
        elif window is not None:
            length = window.shape[0]
        else:
            length = signal.shape[-2]
        if window is None:
            window = torch.ones(length, dtype=signal.dtype, device=signal.device)

        n_frames = 1 + (signal.shape[-2] - length) // step
        frames = signal.unfold(-2, window.shape[0], step).transpose(-1, -2)
        frames = frames.narrow(-3, 0, n_frames) * window.reshape(-1, 1)

        if frames.shape[-1] == 1:
            complex_frames = frames[..., 0]
        else:
            complex_frames = torch.complex(frames[..., 0], frames[..., 1])

        result = torch.fft.fft(complex_frames, n=length, dim=-1)
        output = torch.stack((result.real, result.imag), dim=-1)

        if self.onesided:
            output = output.narrow(-2, 0, output.shape[-2] // 2 + 1)
        return output.to(signal.dtype)
