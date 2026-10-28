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


def prepare_layout_input(layout, input, states):
    """Transpose layout-1 (batch first) tensors into the seq-first view."""
    if layout != 1:
        return input, states
    input = input.transpose(0, 1).contiguous()
    prepared = []
    for state in states:
        if state is None or state.numel() == 0:
            prepared.append(state)
        else:
            # (batch, num_directions, hidden) -> (num_directions, batch, hidden)
            prepared.append(state.transpose(0, 1).contiguous())
    return input, prepared


def prepare_layout_output(layout, output, states):
    """Transpose seq-first recurrent outputs back to layout 1."""
    if layout != 1:
        return output, states
    # (seq, num_directions, batch, hidden) -> (batch, seq, num_directions, hidden)
    output = output.permute(2, 0, 1, 3).contiguous()
    prepared = [state.transpose(0, 1).contiguous() for state in states]
    return output, prepared


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
