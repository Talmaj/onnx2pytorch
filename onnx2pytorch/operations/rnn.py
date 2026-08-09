import torch
from torch import nn


class RNNWrapper(nn.Module):
    """Wraps a 1-layer nn.RNN to match the API of an ONNX RNN.

    It expects h_0 as a separate input rather than as a tuple,
    and returns h_n as a separate output rather than as a tuple.
    """

    def __init__(self, rnn_module, reverse=False):
        super().__init__()
        self.rnn = rnn_module
        self.reverse = reverse

    def forward(self, input, h_0=None):
        (seq_len, batch, input_size) = input.shape
        num_directions = self.rnn.bidirectional + 1
        hidden_size = self.rnn.hidden_size

        if h_0 is not None and h_0.numel() == 0:
            h_0 = None

        if self.reverse:
            input = input.flip(0)
        output, h_n = self.rnn(input, h_0)
        if self.reverse:
            output = output.flip(0)

        # Y has shape (seq_length, num_directions, batch_size, hidden_size)
        Y = output.view(seq_len, batch, num_directions, hidden_size).transpose(1, 2)
        # Y_h has shape (num_directions, batch_size, hidden_size)
        Y_h = h_n.view(1, num_directions, batch, hidden_size).squeeze(0)

        return Y, Y_h
