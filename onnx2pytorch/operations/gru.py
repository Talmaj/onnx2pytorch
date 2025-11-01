import torch
from torch import nn


class GRUWrapper(nn.Module):
    """Wraps a 1-layer nn.GRU or custom GRU to match the API of an ONNX GRU.

    It expects h_0 as a separate input rather than as a tuple,
    and returns h_n as a separate output rather than as a tuple.

    Supports both linear_before_reset=0 and linear_before_reset=1.
    """

    def __init__(self, gru_module, linear_before_reset=1):
        super().__init__()
        self.gru = gru_module
        self.linear_before_reset = linear_before_reset

        # For linear_before_reset=0, we need custom forward pass
        if linear_before_reset == 0 and isinstance(gru_module, nn.GRU):
            # Extract parameters from PyTorch GRU for custom implementation
            self.input_size = gru_module.input_size
            self.hidden_size = gru_module.hidden_size
            self.bidirectional = gru_module.bidirectional

    def forward(self, input, h_0=None):
        (seq_len, batch, input_size) = input.shape
        num_layers = 1
        num_directions = (
            self.gru.bidirectional + 1 if hasattr(self.gru, "bidirectional") else 1
        )
        hidden_size = self.gru.hidden_size

        if h_0 is None or h_0.numel() == 0:
            h_0 = None

        if self.linear_before_reset == 1:
            # Use standard PyTorch GRU (linear_before_reset=1 is PyTorch's default)
            output, h_n = self.gru(input, h_0)
        else:
            # Custom implementation for linear_before_reset=0
            output, h_n = self._forward_linear_before_reset_0(input, h_0)

        # Y has shape (seq_length, num_directions, batch_size, hidden_size)
        Y = output.view(seq_len, batch, num_directions, hidden_size).transpose(1, 2)
        # Y_h has shape (num_directions, batch_size, hidden_size)
        Y_h = h_n.view(num_layers, num_directions, batch, hidden_size).squeeze(0)

        return Y, Y_h

    def _forward_linear_before_reset_0(self, input, h_0):
        """Custom GRU forward with linear_before_reset=0 (ONNX/TensorFlow default).

        Key difference from linear_before_reset=1 (PyTorch default):
        - linear_before_reset=0: ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
          Reset gate is applied to hidden state BEFORE matrix multiplication.
        - linear_before_reset=1: ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
          Reset gate is applied AFTER matrix multiplication and bias addition.

        Equations for linear_before_reset=0:
        r_t = sigmoid(W_ir @ x_t + b_ir + W_hr @ h_{t-1} + b_hr)
        z_t = sigmoid(W_iz @ x_t + b_iz + W_hz @ h_{t-1} + b_hz)
        n_t = tanh(W_in @ x_t + b_in + (r_t * h_{t-1}) @ W_hn + b_hn)
        h_t = (1 - z_t) * n_t + z_t * h_{t-1}
        """
        seq_len, batch, input_size = input.shape
        hidden_size = self.hidden_size
        num_directions = 2 if self.bidirectional else 1

        if h_0 is None:
            h_0 = torch.zeros(
                num_directions,
                batch,
                hidden_size,
                device=input.device,
                dtype=input.dtype,
            )

        # Extract weights from PyTorch GRU
        # PyTorch stores weights as: weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        # For bidirectional: also weight_ih_l0_reverse, weight_hh_l0_reverse, etc.

        def gru_cell_linear_before_reset_0(
            x_t, h_prev, weight_ih, weight_hh, bias_ih, bias_hh
        ):
            """Single GRU cell with linear_before_reset=0."""
            # Split weights for gates: reset, update, new
            # PyTorch order: [reset, update, new]
            hidden_size = h_prev.size(1)

            # Input-to-hidden weights
            W_ir, W_iz, W_in = weight_ih.chunk(3, 0)
            # Hidden-to-hidden weights
            W_hr, W_hz, W_hn = weight_hh.chunk(3, 0)
            # Input biases
            b_ir, b_iz, b_in = (
                bias_ih.chunk(3, 0) if bias_ih is not None else (None, None, None)
            )
            # Hidden biases
            b_hr, b_hz, b_hn = (
                bias_hh.chunk(3, 0) if bias_hh is not None else (None, None, None)
            )

            # Reset gate
            r_t = torch.sigmoid(
                x_t @ W_ir.t()
                + (b_ir if b_ir is not None else 0)
                + h_prev @ W_hr.t()
                + (b_hr if b_hr is not None else 0)
            )

            # Update gate
            z_t = torch.sigmoid(
                x_t @ W_iz.t()
                + (b_iz if b_iz is not None else 0)
                + h_prev @ W_hz.t()
                + (b_hz if b_hz is not None else 0)
            )

            # New gate (linear_before_reset=0 version)
            # Note: Reset gate is applied to h_prev BEFORE matrix multiplication
            # ONNX spec: ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
            n_t = torch.tanh(
                x_t @ W_in.t()
                + (b_in if b_in is not None else 0)
                + (r_t * h_prev) @ W_hn.t()
                + (b_hn if b_hn is not None else 0)
            )

            # Hidden state update
            h_t = (1 - z_t) * n_t + z_t * h_prev

            return h_t

        # Process sequence
        outputs_forward = []
        h_forward = h_0[0]

        for t in range(seq_len):
            h_forward = gru_cell_linear_before_reset_0(
                input[t],
                h_forward,
                self.gru.weight_ih_l0,
                self.gru.weight_hh_l0,
                self.gru.bias_ih_l0 if self.gru.bias else None,
                self.gru.bias_hh_l0 if self.gru.bias else None,
            )
            outputs_forward.append(h_forward)

        if self.bidirectional:
            # Process backward direction
            outputs_backward = []
            h_backward = h_0[1]

            for t in range(seq_len - 1, -1, -1):
                h_backward = gru_cell_linear_before_reset_0(
                    input[t],
                    h_backward,
                    self.gru.weight_ih_l0_reverse,
                    self.gru.weight_hh_l0_reverse,
                    self.gru.bias_ih_l0_reverse if self.gru.bias else None,
                    self.gru.bias_hh_l0_reverse if self.gru.bias else None,
                )
                outputs_backward.append(h_backward)

            outputs_backward.reverse()

            # Concatenate forward and backward outputs
            output = torch.stack(
                [
                    torch.cat([outputs_forward[t], outputs_backward[t]], dim=1)
                    for t in range(seq_len)
                ]
            )
            h_n = torch.stack([h_forward, h_backward])
        else:
            output = torch.stack(outputs_forward)
            h_n = h_forward.unsqueeze(0)

        return output, h_n
