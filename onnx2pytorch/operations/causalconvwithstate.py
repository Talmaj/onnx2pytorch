import torch
from torch import nn
from torch.nn import functional as F


class CausalConvWithState(nn.Module):
    def __init__(self, activation="none"):
        super().__init__()
        if activation not in ("none", "silu", "swish"):
            raise ValueError(
                "CausalConvWithState activation={} not supported.".format(activation)
            )
        self.activation = activation

    def forward(
        self,
        input: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor = None,
        past_state: torch.Tensor = None,
    ):
        batch_size, channels, _ = input.shape
        kernel_size = weight.shape[2]

        if past_state is None:
            past_state = torch.zeros(
                batch_size,
                channels,
                kernel_size - 1,
                dtype=input.dtype,
                device=input.device,
            )
        padded = torch.cat((past_state, input), dim=2)

        output = F.conv1d(padded, weight, bias, groups=channels)
        if self.activation in ("silu", "swish"):
            output = F.silu(output)

        present_state = padded[:, :, padded.shape[2] - (kernel_size - 1) :]
        return output, present_state

    def extra_repr(self) -> str:
        return "activation={}".format(self.activation)
