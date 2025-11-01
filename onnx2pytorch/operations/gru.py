from torch import nn


class GRUWrapper(nn.Module):
    """Wraps a 1-layer nn.GRU to match the API of an ONNX GRU.

    It expects h_0 as a separate input rather than as a tuple,
    and returns h_n as a separate output rather than as a tuple.
    """

    def __init__(self, gru_module: nn.GRU):
        super().__init__()
        self.gru = gru_module

    def forward(self, input, h_0=None):
        (seq_len, batch, input_size) = input.shape
        num_layers = 1
        num_directions = self.gru.bidirectional + 1
        hidden_size = self.gru.hidden_size

        if h_0 is None or h_0.numel() == 0:
            h_0 = None

        output, h_n = self.gru(input, h_0)

        # Y has shape (seq_length, num_directions, batch_size, hidden_size)
        Y = output.view(seq_len, batch, num_directions, hidden_size).transpose(1, 2)
        # Y_h has shape (num_directions, batch_size, hidden_size)
        Y_h = h_n.view(num_layers, num_directions, batch, hidden_size).squeeze(0)

        return Y, Y_h
