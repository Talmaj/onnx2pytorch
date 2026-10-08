from torch import nn

from onnx2pytorch.operations.rnnutils import (
    prepare_layout_input,
    prepare_layout_output,
    reverse_sequences,
    run_padded,
)


class LSTMWrapper(nn.Module):
    """Wraps a 1-layer nn.LSTM to match the API of an ONNX LSTM.

    It expects h_0 and c_0 as separate inputs rather than as a tuple,
    and returns h_n and c_n as separate outputs rather than as a tuple.
    """

    def __init__(
        self,
        lstm_module: nn.LSTM,
        reverse=False,
        sequence_lens=None,
        layout=0,
        initial_h=None,
        initial_c=None,
    ):
        super().__init__()
        self.lstm = lstm_module
        self.reverse = reverse
        self.layout = layout
        self.register_buffer(
            "sequence_lens", None if sequence_lens is None else sequence_lens.long()
        )
        self.register_buffer("initial_h", initial_h)
        self.register_buffer("initial_c", initial_c)

    def forward(self, input, h_0=None, c_0=None):
        if h_0 is None or h_0.numel() == 0:
            h_0 = self.initial_h
        if c_0 is None or c_0.numel() == 0:
            c_0 = self.initial_c
        input, (h_0, c_0) = prepare_layout_input(self.layout, input, (h_0, c_0))
        seq_len, batch, input_size = input.shape
        num_layers = 1
        num_directions = self.lstm.bidirectional + 1
        hidden_size = self.lstm.hidden_size
        lengths = self.sequence_lens
        if (h_0 is None or h_0.numel() == 0) and (c_0 is None or c_0.numel() == 0):
            tuple_0 = None
        else:
            if h_0 is None or h_0.numel() == 0:
                h_0 = input.new_zeros(num_directions, batch, hidden_size)
            if c_0 is None or c_0.numel() == 0:
                c_0 = input.new_zeros(num_directions, batch, hidden_size)
            tuple_0 = (h_0, c_0)
        if self.reverse:
            input = reverse_sequences(input, lengths)
        if lengths is None:
            output, (h_n, c_n) = self.lstm(input, tuple_0)
        else:
            output, (h_n, c_n) = run_padded(self.lstm, input, tuple_0, lengths)
        if self.reverse:
            output = reverse_sequences(output, lengths)

        # Y has shape (seq_length, num_directions, batch_size, hidden_size)
        Y = output.view(seq_len, batch, num_directions, hidden_size).transpose(1, 2)
        # Y_h has shape (num_directions, batch_size, hidden_size)
        Y_h = h_n.view(num_layers, num_directions, batch, hidden_size).squeeze(0)
        # Y_c has shape (num_directions, batch_size, hidden_size)
        Y_c = c_n.view(num_layers, num_directions, batch, hidden_size).squeeze(0)
        Y, (Y_h, Y_c) = prepare_layout_output(self.layout, Y, (Y_h, Y_c))
        return Y, Y_h, Y_c
