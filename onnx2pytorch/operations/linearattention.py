import math

import torch
from torch import nn

UPDATE_RULES = ("linear", "gated", "delta", "gated_delta")


def unpack_heads(x: torch.Tensor, num_heads: int):
    """Reshape (batch, seq, num_heads * dim) to (batch, num_heads, seq, dim)."""
    batch_size, sequence_length, hidden_size = x.shape
    x = x.reshape(batch_size, sequence_length, num_heads, hidden_size // num_heads)
    return x.permute(0, 2, 1, 3)


class LinearAttention(nn.Module):
    def __init__(
        self,
        chunk_size=None,
        kv_num_heads=None,
        q_num_heads=None,
        scale=None,
        update_rule="gated_delta",
    ):
        super().__init__()
        if update_rule not in UPDATE_RULES:
            raise ValueError(
                "LinearAttention update_rule={} not supported.".format(update_rule)
            )
        # chunk_size is a tuning hint without effect on the result.
        self.chunk_size = chunk_size
        self.kv_num_heads = kv_num_heads
        self.q_num_heads = q_num_heads
        self.scale = scale
        self.update_rule = update_rule

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        past_state: torch.Tensor = None,
        decay: torch.Tensor = None,
        beta: torch.Tensor = None,
    ):
        gating = self.update_rule in ("gated", "gated_delta")
        delta_correction = self.update_rule in ("delta", "gated_delta")

        batch_size, sequence_length, _ = query.shape
        q_num_heads = self.q_num_heads
        kv_num_heads = self.kv_num_heads
        key_dim = query.shape[-1] // q_num_heads
        value_dim = value.shape[-1] // kv_num_heads
        group_size = q_num_heads // kv_num_heads

        q4 = unpack_heads(query, q_num_heads).float()
        k4 = unpack_heads(key, kv_num_heads).float()
        v4 = unpack_heads(value, kv_num_heads).float()

        if gating:
            if decay.shape[-1] == kv_num_heads:
                decay4 = decay.reshape(
                    batch_size, sequence_length, kv_num_heads, 1
                ).permute(0, 2, 1, 3)
            else:
                decay4 = unpack_heads(decay, kv_num_heads)
            decay4 = decay4.float()
        if delta_correction:
            beta4 = beta.reshape(
                batch_size, sequence_length, beta.shape[-1], 1
            ).permute(0, 2, 1, 3)
            beta4 = beta4.float()

        if past_state is None:
            state_dtype = query.dtype
            state = torch.zeros(
                batch_size,
                kv_num_heads,
                key_dim,
                value_dim,
                dtype=torch.float32,
                device=query.device,
            )
        else:
            state_dtype = past_state.dtype
            state = past_state.float().clone()

        scale = self.scale or 1 / math.sqrt(key_dim)

        outputs = []
        for i in range(sequence_length):
            q_t = q4[:, :, i, :]
            k_t = k4[:, :, i, :]
            v_t = v4[:, :, i, :]

            if gating:
                state = state * torch.exp(decay4[:, :, i, :]).unsqueeze(-1)
            if delta_correction:
                retrieved = torch.einsum("bhdm,bhd->bhm", state, k_t)
                v_t = beta4[:, :, i, :] * (v_t - retrieved)
            state = state + k_t.unsqueeze(-1) * v_t.unsqueeze(-2)

            read_state = (
                state if group_size == 1 else state.repeat_interleave(group_size, dim=1)
            )
            outputs.append(scale * torch.einsum("bhd,bhdm->bhm", q_t, read_state))

        output = torch.stack(outputs, dim=2).permute(0, 2, 1, 3)
        output = output.reshape(
            batch_size, sequence_length, q_num_heads * value_dim
        ).to(query.dtype)
        return output, state.to(state_dtype)

    def extra_repr(self) -> str:
        return "update_rule={}, q_num_heads={}, kv_num_heads={}, scale={}".format(
            self.update_rule, self.q_num_heads, self.kv_num_heads, self.scale
        )
