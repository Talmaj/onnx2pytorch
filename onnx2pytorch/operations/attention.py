import math

import torch
from torch import nn
from torch.nn import functional as F

from onnx2pytorch.dtypes import ONNX_DTYPE_TO_TORCH

NEG_INF = float("-inf")


def split_heads(x: torch.Tensor, num_heads: int):
    """Reshape (batch, seq, num_heads * head_size) to (batch, num_heads, seq, head_size)."""
    batch_size, sequence_length, hidden_size = x.shape
    x = x.reshape(batch_size, sequence_length, num_heads, hidden_size // num_heads)
    return x.permute(0, 2, 1, 3)


def causal_bias(base: torch.Tensor, offset, q_length, kv_length):
    """
    Add an offset-aligned (bottom-right) causal bias to base.

    A query at index i attends key j iff j <= i + offset. The offset is either a
    scalar or, for a per-batch external cache, a 1D tensor of batch size.
    """
    i = torch.arange(q_length, device=base.device).reshape(q_length, 1)
    j = torch.arange(kv_length, device=base.device).reshape(1, kv_length)
    per_batch = torch.is_tensor(offset) and offset.dim() > 0
    if per_batch:
        allowed = j <= (i + offset.reshape(-1, 1, 1))
    else:
        allowed = j <= (i + int(offset))
    causal = torch.zeros(allowed.shape, dtype=base.dtype, device=base.device)
    causal.masked_fill_(~allowed, NEG_INF)
    if per_batch:
        base = base.reshape((1,) * (4 - base.dim()) + tuple(base.shape))
        return base + causal.reshape(-1, 1, q_length, kv_length)
    return base + causal


def bool_mask_to_bias(mask: torch.Tensor, dtype):
    bias = torch.zeros(mask.shape, dtype=dtype, device=mask.device)
    bias.masked_fill_(~mask, NEG_INF)
    return bias


class Attention(nn.Module):
    def __init__(
        self,
        is_causal=0,
        kv_num_heads=None,
        q_num_heads=None,
        qk_matmul_output_mode=0,
        scale=None,
        softcap=0.0,
        softmax_precision=None,
        num_outputs=1,
    ):
        super().__init__()
        self.is_causal = bool(is_causal)
        self.kv_num_heads = kv_num_heads
        self.q_num_heads = q_num_heads
        self.qk_matmul_output_mode = qk_matmul_output_mode
        self.scale = scale
        self.softcap = softcap or 0.0
        self.softmax_precision = softmax_precision
        self.num_outputs = num_outputs

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        attn_mask: torch.Tensor = None,
        past_key: torch.Tensor = None,
        past_value: torch.Tensor = None,
        nonpad_kv_seqlen: torch.Tensor = None,
    ):
        input_rank = Q.dim()
        batch_size = Q.shape[0]
        if input_rank == 3:
            Q = split_heads(Q, self.q_num_heads)
            K = split_heads(K, self.kv_num_heads)
            V = split_heads(V, self.kv_num_heads)

        scale = self.scale
        if scale is None:
            scale = 1 / math.sqrt(Q.shape[3])

        present_key = K if past_key is None else torch.cat((past_key, K), dim=2)
        present_value = V if past_value is None else torch.cat((past_value, V), dim=2)
        K, V = present_key, present_value

        q_length = Q.shape[2]
        kv_length = K.shape[2]

        # The mask may be shorter than the (cache-extended) key length.
        if attn_mask is not None:
            pad_width = kv_length - attn_mask.shape[-1]
            if pad_width > 0:
                pad_value = False if attn_mask.dtype == torch.bool else NEG_INF
                attn_mask = F.pad(attn_mask, (0, pad_width), value=pad_value)
            if attn_mask.dtype == torch.bool:
                attn_mask = bool_mask_to_bias(attn_mask, Q.dtype)
            else:
                attn_mask = attn_mask.to(Q.dtype)

        attn_bias = torch.zeros(q_length, kv_length, dtype=Q.dtype, device=Q.device)
        if self.is_causal:
            base = attn_bias if attn_mask is None else attn_mask
            if past_key is None and nonpad_kv_seqlen is not None:
                offset = nonpad_kv_seqlen.reshape(-1) - q_length
            else:
                offset = 0 if past_key is None else past_key.shape[2]
            attn_bias = causal_bias(base, offset, q_length, kv_length)
        elif attn_mask is not None:
            attn_bias = attn_bias + attn_mask

        if nonpad_kv_seqlen is not None:
            attn_bias = attn_bias.reshape(
                (1,) * (4 - attn_bias.dim()) + tuple(attn_bias.shape)
            )
            padding_mask = torch.arange(
                kv_length, device=Q.device
            ) < nonpad_kv_seqlen.reshape(-1, 1)
            padding_mask = padding_mask.reshape(batch_size, 1, 1, kv_length)
            attn_bias = attn_bias + bool_mask_to_bias(padding_mask, Q.dtype)

        # Group query attention replicates each key/value head across its group.
        q_num_heads = self.q_num_heads or Q.shape[1]
        kv_num_heads = self.kv_num_heads or K.shape[1]
        if q_num_heads != kv_num_heads and q_num_heads % kv_num_heads == 0:
            repeats = q_num_heads // kv_num_heads
            K = K.repeat_interleave(repeats, dim=1)
            V = V.repeat_interleave(repeats, dim=1)

        # A query row whose bias is entirely -inf has no attendable key and
        # softmaxes to all zeros instead of NaN.
        row_all_masked = torch.isneginf(attn_bias.amax(dim=-1, keepdim=True))

        qk_matmul_output = None
        # scaled_dot_product_attention softmaxes in the input dtype, so an
        # explicit softmax_precision needs the manual path
        if (
            self.num_outputs < 4
            and self.softcap <= 0
            and self.softmax_precision is None
        ):
            y = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=attn_bias, scale=scale
            )
            y = torch.where(row_all_masked, torch.zeros_like(y), y)
        else:
            root_scale = math.sqrt(scale)
            qk = torch.matmul(Q * root_scale, K.transpose(-1, -2) * root_scale)
            # Softcap goes before the bias, so that a -inf stays -inf.
            capped = qk
            if self.softcap > 0:
                capped = torch.tanh(qk / self.softcap) * self.softcap
            qk_with_bias = capped + attn_bias
            if self.qk_matmul_output_mode == 2:
                qk_matmul_output = qk_with_bias
            elif self.qk_matmul_output_mode == 1:
                qk_matmul_output = capped
            else:
                qk_matmul_output = qk
            if self.softmax_precision is not None:
                dtype = ONNX_DTYPE_TO_TORCH.get(self.softmax_precision)
                if dtype is None:
                    raise ValueError(
                        "Attention softmax_precision {} is not supported in "
                        "pytorch.".format(self.softmax_precision)
                    )
                qk_with_bias = qk_with_bias.to(dtype)
            probs = torch.softmax(qk_with_bias, dim=-1)
            probs = torch.where(row_all_masked, torch.zeros_like(probs), probs)
            if self.qk_matmul_output_mode == 3:
                qk_matmul_output = probs
            qk_matmul_output = qk_matmul_output.to(Q.dtype)
            y = torch.matmul(probs.to(V.dtype), V).to(Q.dtype)

        if input_rank == 3:
            y = y.permute(0, 2, 1, 3).reshape(batch_size, q_length, -1)

        outputs = [y, present_key, present_value, qk_matmul_output]
        return outputs[: self.num_outputs]

    def extra_repr(self) -> str:
        return (
            "is_causal={}, q_num_heads={}, kv_num_heads={}, scale={}, softcap={}, "
            "qk_matmul_output_mode={}".format(
                self.is_causal,
                self.q_num_heads,
                self.kv_num_heads,
                self.scale,
                self.softcap,
                self.qk_matmul_output_mode,
            )
        )
