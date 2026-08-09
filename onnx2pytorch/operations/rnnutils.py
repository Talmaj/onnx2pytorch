"""Helpers shared by the ONNX RNN, GRU and LSTM wrappers."""

import torch
from torch import nn


def step_mask(lengths, seq_len, dtype):
    """(seq_len, batch) mask that is one for the steps inside each sequence."""
    steps = torch.arange(seq_len, device=lengths.device).unsqueeze(1)
    return (steps < lengths.unsqueeze(0)).to(dtype)


def reverse_sequences(input, lengths):
    """Reverse each sequence over its own length, leaving the padding in place."""
    if lengths is None:
        return input.flip(0)
    seq_len = input.shape[0]
    steps = torch.arange(seq_len, device=input.device).unsqueeze(1)
    lens = lengths.unsqueeze(0)
    index = torch.where(steps < lens, (lens - 1 - steps).clamp(min=0), steps)
    index = index.view(seq_len, -1, *([1] * (input.dim() - 2))).expand_as(input)
    return input.gather(0, index)


def run_padded(module, input, state, lengths):
    """Run a torch RNN module so that steps beyond each sequence length are dropped.

    Zero lengths have no packed representation, so they are run for one step and
    masked out again, which leaves the output and the final state at zero.
    """
    seq_len = input.shape[0]
    packed = nn.utils.rnn.pack_padded_sequence(
        input, lengths.clamp(min=1).cpu(), enforce_sorted=False
    )
    output, state_n = module(packed, state)
    output, _ = nn.utils.rnn.pad_packed_sequence(output, total_length=seq_len)
    output = output * step_mask(lengths, seq_len, output.dtype).unsqueeze(-1)
    keep = (lengths > 0).view(1, -1, 1)
    if isinstance(state_n, tuple):
        state_n = tuple(s * keep.to(s.dtype) for s in state_n)
    else:
        state_n = state_n * keep.to(state_n.dtype)
    return output, state_n
