"""ONNX RNN, GRU and LSTM for the cases nn.RNN cannot represent.

The default activations go through nn.RNN / nn.GRU / nn.LSTM. This module is the
step-by-step fallback used when the node asks for something those modules do
not implement: a non-default activation, activation_alpha / activation_beta,
clip, LSTM peepholes or a coupled input/forget gate.
"""

import torch
from torch import nn

from onnx2pytorch.operations.rnnutils import prepare_layout_input, prepare_layout_output

_ACTIVATION_COUNT = {"RNN": 1, "GRU": 2, "LSTM": 3}
_WEIGHT_GATES = {"RNN": 1, "GRU": 3, "LSTM": 4}
_DEFAULT_ACTIVATIONS = {
    "RNN": ("Tanh",),
    "GRU": ("Sigmoid", "Tanh"),
    "LSTM": ("Sigmoid", "Tanh", "Tanh"),
}


def _apply_activation(name, value, alpha, beta):
    """One ONNX RNN activation. alpha/beta are None when the node omits them."""
    key = name.lower().replace("_", "")
    if key == "relu":
        return torch.relu(value)
    if key == "tanh":
        return torch.tanh(value)
    if key == "sigmoid":
        return torch.sigmoid(value)
    if key == "affine":
        scale = 1.0 if alpha is None else alpha
        shift = 0.0 if beta is None else beta
        return scale * value + shift
    if key == "leakyrelu":
        return torch.nn.functional.leaky_relu(value, 0.01 if alpha is None else alpha)
    if key == "thresholdedrelu":
        threshold = 1.0 if alpha is None else alpha
        return torch.where(value >= threshold, value, torch.zeros_like(value))
    if key == "scaledtanh":
        scale = 1.0 if alpha is None else alpha
        inner = 1.0 if beta is None else beta
        return scale * torch.tanh(inner * value)
    if key == "hardsigmoid":
        scale = 0.2 if alpha is None else alpha
        shift = 0.5 if beta is None else beta
        return torch.clamp(scale * value + shift, 0, 1)
    if key == "elu":
        return torch.nn.functional.elu(value, 1.0 if alpha is None else alpha)
    if key == "softsign":
        return value / (1 + value.abs())
    if key == "softplus":
        return torch.nn.functional.softplus(value)
    raise NotImplementedError("activations {}".format(name))


def _direction_activations(op_type, activations, alpha, beta, bidirectional):
    """Per-direction (name, alpha, beta) triples, in ONNX gate order."""
    count = _ACTIVATION_COUNT[op_type]
    if activations is None:
        names = list(_DEFAULT_ACTIVATIONS[op_type])
        if bidirectional:
            names = names + names
    else:
        names = list(activations)
    expected = count * (2 if bidirectional else 1)
    shared = len(names) == count
    if len(names) != expected and not (bidirectional and shared):
        raise NotImplementedError(
            "{} activations {}.".format(op_type, list(activations or names))
        )
    alphas = list(alpha or [])
    betas = list(beta or [])

    def specs(offset, used_names):
        triples = []
        for i, name in enumerate(used_names):
            index = i if shared else offset + i
            a = alphas[index] if index < len(alphas) else None
            b = betas[index] if index < len(betas) else None
            # Fail at conversion time, not on the first forward.
            try:
                _apply_activation(name, torch.zeros(1), a, b)
            except NotImplementedError:
                shown = list(activations) if activations is not None else list(names)
                raise NotImplementedError("{} activations {}.".format(op_type, shown))
            triples.append((name, a, b))
        return triples

    if shared:
        one = specs(0, names[:count])
        return [one, one] if bidirectional else [one]
    return [specs(0, names[:count]), specs(count, names[count:])]


class OnnxRecurrent(nn.Module):
    """Single-layer ONNX RNN, GRU or LSTM evaluated from the spec equations."""

    def __init__(
        self,
        op_type,
        weight_ih,
        weight_hh,
        bias,
        peephole,
        hidden_size,
        direction="forward",
        activations=None,
        activation_alpha=None,
        activation_beta=None,
        clip=None,
        layout=0,
        linear_before_reset=0,
        input_forget=0,
        sequence_lens=None,
        initial_h=None,
        initial_c=None,
    ):
        super().__init__()
        if op_type not in _ACTIVATION_COUNT:
            raise ValueError("Unknown recurrent op {}.".format(op_type))
        if direction not in ("forward", "reverse", "bidirectional"):
            raise ValueError("{} direction {}.".format(op_type, direction))
        if layout not in (0, 1):
            raise NotImplementedError("{} layout={}.".format(op_type, layout))
        if input_forget not in (0, 1):
            raise NotImplementedError(
                "{} input_forget {}.".format(op_type, input_forget)
            )

        self.op_type = op_type
        self.hidden_size = hidden_size
        self.direction = direction
        self.clip = clip
        self.layout = layout
        self.linear_before_reset = linear_before_reset
        self.input_forget = input_forget
        self.activations_attr = activations
        bidirectional = direction == "bidirectional"
        self.activation_specs = _direction_activations(
            op_type, activations, activation_alpha, activation_beta, bidirectional
        )

        self.W = nn.Parameter(weight_ih)
        self.R = nn.Parameter(weight_hh)
        if bias is None:
            self.B = None
        else:
            self.B = nn.Parameter(bias)
        if peephole is None:
            self.P = None
        else:
            self.P = nn.Parameter(peephole)
        self.register_buffer(
            "sequence_lens", None if sequence_lens is None else sequence_lens.long()
        )
        self.register_buffer("initial_h", initial_h)
        self.register_buffer("initial_c", initial_c)

    def _clip(self, value):
        # ORT clips the input of each activation, not the hidden or cell state.
        if self.clip is None:
            return value
        return torch.clamp(value, -self.clip, self.clip)

    def _activate(self, direction, index, value):
        name, alpha, beta = self.activation_specs[direction][index]
        try:
            return _apply_activation(name, value, alpha, beta)
        except NotImplementedError:
            shown = self.activations_attr
            if shown is None:
                shown = _DEFAULT_ACTIVATIONS[self.op_type]
            raise NotImplementedError(
                "{} activations {}.".format(self.op_type, list(shown))
            )

    def _bias(self, direction):
        gates = _WEIGHT_GATES[self.op_type]
        hidden = self.hidden_size
        width = gates * hidden
        if self.B is None:
            zero = self.W.new_zeros(width)
            return zero, zero
        return self.B[direction, :width], self.B[direction, width:]

    def _step(self, direction, x_t, h, c):
        hidden = self.hidden_size
        weight_hh = self.R[direction]
        _, recurrent_bias = self._bias(direction)
        if self.op_type == "RNN":
            hidden_state = self._activate(
                direction,
                0,
                self._clip(x_t + h.matmul(weight_hh.t()) + recurrent_bias),
            )
            return hidden_state, None

        if self.op_type == "GRU":
            xz, xr, xh = x_t.split(hidden, dim=-1)
            rz, rr, rh = weight_hh.split(hidden, dim=0)
            bz, br, bh = recurrent_bias.split(hidden, dim=0)
            z = self._activate(direction, 0, self._clip(xz + h.matmul(rz.t()) + bz))
            r = self._activate(direction, 0, self._clip(xr + h.matmul(rr.t()) + br))
            if self.linear_before_reset:
                candidate = xh + r * (h.matmul(rh.t()) + bh)
            else:
                candidate = xh + (r * h).matmul(rh.t()) + bh
            h_hat = self._activate(direction, 1, self._clip(candidate))
            return (1 - z) * h_hat + z * h, None

        xi, xo, xf, xc = x_t.split(hidden, dim=-1)
        ri, ro, rf, rc = weight_hh.split(hidden, dim=0)
        bi, bo, bf, bc = recurrent_bias.split(hidden, dim=0)
        i = xi + h.matmul(ri.t()) + bi
        o = xo + h.matmul(ro.t()) + bo
        f = xf + h.matmul(rf.t()) + bf
        cell_input = xc + h.matmul(rc.t()) + bc
        if self.P is not None:
            pi, po, pf = self.P[direction].split(hidden, dim=0)
            i = i + pi * c
            f = f + pf * c
        i = self._activate(direction, 0, self._clip(i))
        if self.input_forget:
            f = 1 - i
        else:
            f = self._activate(direction, 0, self._clip(f))
        cell = f * c + i * self._activate(direction, 1, self._clip(cell_input))
        if self.P is not None:
            # The output peephole sees the updated cell, then the sum is clipped.
            o = o + po * cell
        o = self._activate(direction, 0, self._clip(o))
        hidden_state = o * self._activate(direction, 2, cell)
        return hidden_state, cell

    def _run_direction(self, x, h, c, direction, reverse, lengths):
        seq_len, batch, _ = x.shape
        hidden = self.hidden_size
        input_bias, _ = self._bias(direction)
        x_lin = x.matmul(self.W[direction].t()) + input_bias
        if c is None:
            c = x.new_zeros(batch, hidden)
        outputs = [None] * seq_len
        steps = range(seq_len - 1, -1, -1) if reverse else range(seq_len)
        for t in steps:
            h_new, c_new = self._step(direction, x_lin[t], h, c)
            if lengths is None:
                h, c = h_new, c_new
                outputs[t] = h_new
                continue
            active = (lengths > t).to(h_new.dtype).unsqueeze(-1)
            h = active * h_new + (1 - active) * h
            if c_new is not None:
                c = active * c_new + (1 - active) * c
            outputs[t] = active * h_new
        if lengths is not None:
            # A zero length has no steps, so the final state is zeros.
            keep = (lengths > 0).to(h.dtype).unsqueeze(-1)
            h = h * keep
            if c is not None:
                c = c * keep
        return torch.stack(outputs, 0), h, c

    def forward(self, input, h_0=None, c_0=None):
        if h_0 is None or h_0.numel() == 0:
            h_0 = self.initial_h
        if c_0 is None or c_0.numel() == 0:
            c_0 = self.initial_c
        input, (h_0, c_0) = prepare_layout_input(self.layout, input, (h_0, c_0))

        seq_len, batch, _ = input.shape
        num_directions = 2 if self.direction == "bidirectional" else 1
        hidden = self.hidden_size
        if h_0 is None or h_0.numel() == 0:
            h_0 = input.new_zeros(num_directions, batch, hidden)
        if self.op_type == "LSTM" and (c_0 is None or c_0.numel() == 0):
            c_0 = input.new_zeros(num_directions, batch, hidden)

        lengths = self.sequence_lens
        if lengths is not None:
            lengths = lengths.to(device=input.device)

        directions = []
        for index in range(num_directions):
            reverse = self.direction == "reverse" or (
                self.direction == "bidirectional" and index == 1
            )
            cell = None if c_0 is None else c_0[index]
            directions.append(
                self._run_direction(input, h_0[index], cell, index, reverse, lengths)
            )

        y = torch.stack([item[0] for item in directions], dim=1)
        y_h = torch.stack([item[1] for item in directions], dim=0)
        y, (y_h,) = prepare_layout_output(self.layout, y, (y_h,))
        if self.op_type != "LSTM":
            return y, y_h
        y_c = torch.stack([item[2] for item in directions], dim=0)
        if self.layout == 1:
            y_c = y_c.transpose(0, 1).contiguous()
        return y, y_h, y_c
