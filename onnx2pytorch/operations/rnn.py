from torch import nn

from onnx2pytorch.operations.rnnutils import reverse_sequences, run_padded


class RNNWrapper(nn.Module):
    """Wraps a 1-layer nn.RNN to match the API of an ONNX RNN.

    It expects h_0 as a separate input rather than as a tuple,
    and returns h_n as a separate output rather than as a tuple.
    """

    def __init__(self, rnn_module, reverse=False, sequence_lens=None):
        super().__init__()
        self.rnn = rnn_module
        self.reverse = reverse
        self.register_buffer(
            "sequence_lens", None if sequence_lens is None else sequence_lens.long()
        )

    def forward(self, input, h_0=None):
        (seq_len, batch, input_size) = input.shape
        num_directions = self.rnn.bidirectional + 1
        hidden_size = self.rnn.hidden_size
        lengths = self.sequence_lens

        if h_0 is not None and h_0.numel() == 0:
            h_0 = None

        if self.reverse:
            input = reverse_sequences(input, lengths)
        if lengths is None:
            output, h_n = self.rnn(input, h_0)
        else:
            output, h_n = run_padded(self.rnn, input, h_0, lengths)
        if self.reverse:
            output = reverse_sequences(output, lengths)

        # Y has shape (seq_length, num_directions, batch_size, hidden_size)
        Y = output.view(seq_len, batch, num_directions, hidden_size).transpose(1, 2)
        # Y_h has shape (num_directions, batch_size, hidden_size)
        Y_h = h_n.view(1, num_directions, batch, hidden_size).squeeze(0)

        return Y, Y_h
