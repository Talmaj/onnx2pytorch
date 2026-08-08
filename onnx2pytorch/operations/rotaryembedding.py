import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, interleaved=0, num_heads=0, rotary_embedding_dim=0):
        super().__init__()
        self.interleaved = bool(interleaved)
        self.num_heads = num_heads
        self.rotary_embedding_dim = rotary_embedding_dim

    def forward(
        self,
        input: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        position_ids: torch.Tensor = None,
    ):
        input_shape = input.shape
        if input.dim() == 4:
            # (batch, num_heads, seq, head_size) -> (batch, seq, num_heads, head_size)
            x = input.permute(0, 2, 1, 3)
        else:
            if not self.num_heads:
                raise ValueError(
                    "RotaryEmbedding with 3D input requires the num_heads attribute."
                )
            batch_size, sequence_length, hidden_size = input.shape
            x = input.reshape(
                batch_size,
                sequence_length,
                self.num_heads,
                hidden_size // self.num_heads,
            )

        head_size = x.shape[3]
        rotary_dim = self.rotary_embedding_dim or head_size
        x_rotate = x[:, :, :, :rotary_dim]
        x_not_rotate = x[:, :, :, rotary_dim:]

        if position_ids is not None:
            cos_cache = cos_cache[position_ids.long()]
            sin_cache = sin_cache[position_ids.long()]
        if cos_cache.shape[-1] != rotary_dim // 2:
            raise ValueError(
                "Last dimension of cos/sin cache {} does not match "
                "rotary_embedding_dim/2 {}.".format(
                    cos_cache.shape[-1], rotary_dim // 2
                )
            )
        cos_cache = cos_cache.unsqueeze(2)
        sin_cache = sin_cache.unsqueeze(2)

        if self.interleaved:
            x1 = x_rotate[:, :, :, 0::2]
            x2 = x_rotate[:, :, :, 1::2]
        else:
            x1, x2 = x_rotate.chunk(2, dim=-1)

        real = cos_cache * x1 - sin_cache * x2
        imag = sin_cache * x1 + cos_cache * x2

        if self.interleaved:
            x_rotate = torch.stack((real, imag), dim=-1).reshape(x_rotate.shape)
        else:
            x_rotate = torch.cat((real, imag), dim=-1)

        output = torch.cat((x_rotate, x_not_rotate), dim=-1)
        if len(input_shape) == 3:
            return output.reshape(input_shape)
        return output.permute(0, 2, 1, 3)

    def extra_repr(self) -> str:
        return "interleaved={}, num_heads={}, rotary_embedding_dim={}".format(
            self.interleaved, self.num_heads, self.rotary_embedding_dim
        )
