import numpy as np
import torch
from torch import nn
from onnx import numpy_helper

from onnx2pytorch.operations import (
    AutoPad,
    AveragePool,
    BatchNormWrapper,
    ConvTranspose,
    DilatedAvgPool,
    GRUWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    MaxPool,
    RNNWrapper,
    Transpose,
)
from onnx2pytorch.operations.dynamicconv import DynamicConv
from onnx2pytorch.operations.dynamicgemm import DynamicGemm
from onnx2pytorch.operations.recurrent import OnnxRecurrent
from onnx2pytorch.convert.attribute import extract_attributes, extract_attr_values
from onnx2pytorch.utils import as_tuple


def symmetric_pad_layer(padding, ndim):
    """Turn a symmetric conv-style padding back into a padding layer."""
    pads = [p for p in reversed(as_tuple(padding, ndim)) for _ in range(2)]
    return getattr(nn, "ConstantPad{}d".format(ndim))(pads, value=0)


def extract_params(params):
    """Extract weights and biases."""
    param_length = len(params)
    if param_length == 1:
        weight = params[0]
        bias = None
    elif param_length == 2:
        weight = params[0]
        bias = params[1]
    else:
        raise ValueError("Unexpected number of parameters: {}".format(param_length))
    return weight, bias


def load_params(layer, weight, bias):
    """Load weight and bias to a given layer from onnx format."""
    layer.weight.data = torch.tensor(numpy_helper.to_array(weight))
    if bias is not None:
        layer.bias.data = torch.tensor(numpy_helper.to_array(bias))


def _conv_param(weights, node, index):
    """The initializer/constant at a conv input, or None when it is computed."""
    if weights is None or index >= len(node.input):
        return None
    name = node.input[index]
    if not name or name not in weights:
        return None
    return weights[name]


def _dynamic_conv(
    layer_type, kwargs, pad_layer, transpose_pads, output_shape, auto_pad
):
    kernel_size = kwargs.get("kernel_size")
    ndim = None if kernel_size is None else len(tuple(kernel_size))
    padding = kwargs.get("padding", 0)
    if isinstance(padding, nn.Module):
        padding = 0
    return DynamicConv(
        ndim=ndim,
        stride=kwargs.get("stride", 1),
        padding=padding,
        dilation=kwargs.get("dilation", 1),
        groups=kwargs.get("groups", 1),
        transposed=layer_type == "ConvTranspose",
        output_padding=kwargs.get("output_padding", 0),
        pad_layer=pad_layer,
        transpose_pads=transpose_pads,
        output_shape=output_shape,
        auto_pad=auto_pad,
        kernel_size=kernel_size,
    )


def convert_layer(node, layer_type, params=None, opset_version=None, weights=None):
    """Use to convert Conv, MaxPool, AvgPool layers."""
    assert layer_type in [
        "Conv",
        "ConvTranspose",
        "MaxPool",
        "AvgPool",
    ], "Incorrect layer type: {}".format(layer_type)
    kwargs = extract_attributes(node)
    if layer_type in ("MaxPool", "AvgPool"):
        # ONNX strides default to 1, torch's pooling layers default to kernel_size
        kwargs.setdefault("stride", 1)
    weight_proto = _conv_param(weights, node, 1)
    if weight_proto is None and params and layer_type in ("Conv", "ConvTranspose"):
        weight_proto = params[0]
    if "kernel_size" not in kwargs and weight_proto is not None:
        # kernel_shape is optional for the convolutions, it follows from the weight
        kwargs["kernel_size"] = tuple(weight_proto.dims[2:])
    if kwargs.get("kernel_size") is None and layer_type in ("Conv", "ConvTranspose"):
        kernel_size_length = None
        layer = None
    else:
        kernel_size_length = len(kwargs["kernel_size"])
        try:
            layer = getattr(nn, "{}{}d".format(layer_type, kernel_size_length))
        except AttributeError:
            raise ValueError(
                "Unexpected length of kernel_size dimension: {}".format(
                    kernel_size_length
                )
            )

    storage_order = kwargs.pop("storage_order", 0)

    # Handle auto_pad attribute
    pad_layer = None
    transpose_pads = None
    output_shape = None
    auto_pad = kwargs.pop("auto_pad", "NOTSET")
    if layer_type == "ConvTranspose":
        # ConvTranspose never pads its input, the pads are cropped off the output
        transpose_pads = kwargs.pop("pads", None)
        output_shape = kwargs.pop("output_shape", None)
        kwargs["padding"] = 0
    elif auto_pad == "VALID":
        # No padding
        kwargs["padding"] = 0
    elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        # Create dynamic padding layer
        # Note: explicit pads are ignored when auto_pad is not NOTSET (per ONNX spec)
        if kwargs.get("kernel_size") is None:
            raise NotImplementedError(
                "{} auto_pad={} with a dynamic weight and no kernel_shape.".format(
                    layer_type, auto_pad
                )
            )
        pad_layer = AutoPad(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs.get("stride", 1),
            dilation=kwargs.get("dilation", 1),
            mode=auto_pad,
            # MaxPool ignores the padded cells, the others treat them as zeros
            value=float("-inf") if layer_type == "MaxPool" else 0.0,
        )
        kwargs["padding"] = 0  # Conv layer itself should not pad

    # if padding is a layer, remove from kwargs and prepend later
    if isinstance(kwargs.get("padding"), nn.Module):
        pad_layer = kwargs.pop("padding")

    count_include_pad = True
    dilated_pool = None
    if layer_type == "AvgPool":
        # ONNX excludes the pads from the average since opset 7, torch includes them
        default = opset_version is not None and opset_version < 7
        count_include_pad = bool(kwargs.pop("count_include_pad", default))
        dilation = kwargs.pop("dilation", 1)
        if any(d != 1 for d in as_tuple(dilation, kernel_size_length)):
            if kwargs.pop("ceil_mode", False):
                raise NotImplementedError(
                    "AveragePool with both dilations and ceil_mode not implemented."
                )
            padding = kwargs.pop("padding", 0)
            if pad_layer is None and any(as_tuple(padding, kernel_size_length)):
                pad_layer = symmetric_pad_layer(padding, kernel_size_length)
            dilated_pool = DilatedAvgPool(
                kwargs["kernel_size"], kwargs.get("stride", 1), dilation
            )
        else:
            # A materialised pad reaches the pool as a value, corrected for below
            kwargs["count_include_pad"] = pad_layer is not None or count_include_pad

    weight_dynamic = (
        layer_type in ("Conv", "ConvTranspose")
        and weights is not None
        and _conv_param(weights, node, 1) is None
    )
    bias_name = node.input[2] if len(node.input) > 2 else ""
    bias_dynamic = (
        layer_type in ("Conv", "ConvTranspose")
        and weights is not None
        and bool(bias_name)
        and _conv_param(weights, node, 2) is None
    )
    if weight_dynamic or bias_dynamic:
        # The weight tensor is an input of this module, resolved at runtime.
        return _dynamic_conv(
            layer_type, kwargs, pad_layer, transpose_pads, output_shape, auto_pad
        )
    if layer_type in ("Conv", "ConvTranspose") and weights is not None:
        static_weight = _conv_param(weights, node, 1)
        static_bias = _conv_param(weights, node, 2)
        params = [param for param in (static_weight, static_bias) if param is not None]

    if dilated_pool is not None:
        layer = dilated_pool
    elif params:
        weight, bias = extract_params(params)
        kwargs["bias"] = bias is not None
        kwargs["in_channels"] = weight.dims[1] * kwargs.get("groups", 1)
        kwargs["out_channels"] = weight.dims[0]

        if layer_type == "ConvTranspose":
            kwargs["in_channels"], kwargs["out_channels"] = (
                kwargs["out_channels"],
                kwargs["in_channels"],
            )

        # initialize layer and load weights
        layer = layer(**kwargs)
        load_params(layer, weight, bias)
    else:
        # initialize operations without parameters (MaxPool, AvgPool, etc.)
        if layer_type == "MaxPool":
            kwargs["return_indices"] = True
        layer = layer(**kwargs)

    if layer_type == "ConvTranspose":
        layer = ConvTranspose(layer, transpose_pads, output_shape, auto_pad)

    if layer_type == "AvgPool" and pad_layer is not None and not count_include_pad:
        # The pad is part of the tensor, so divide by the real elements per window
        layer = AveragePool(layer, pad_layer)
        pad_layer = None

    if layer_type == "MaxPool" and len(node.output) > 1:
        # The wrapper pads itself, it has to undo the pad offset in the indices
        layer = MaxPool(layer, storage_order, pad_layer)
        pad_layer = None

    if pad_layer is not None:
        layer = nn.Sequential(pad_layer, layer)

    return layer


def convert_batch_norm_layer(node, params):
    kwargs = extract_attributes(node)
    # Skip input dimension check, not possible before forward pass
    layer = BatchNormWrapper
    torch_params = [torch.tensor(numpy_helper.to_array(param)) for param in params]

    # Initialize layer and load weights
    layer = layer(torch_params, **kwargs)
    return layer


def convert_instance_norm_layer(node, params):
    kwargs = extract_attributes(node)
    # Skip input dimension check, not possible before forward pass
    layer = InstanceNormWrapper
    torch_params = [torch.tensor(numpy_helper.to_array(param)) for param in params]

    # Initialize layer and load weights
    layer = layer(torch_params, **kwargs)
    return layer


def convert_linear_layer(node, params):
    """Convert linear layer from onnx node and params."""
    # Default Gemm attributes
    dc = dict(
        transpose_weight=True,
        transpose_activation=False,
        weight_multiplier=1,
        bias_multiplier=1,
    )
    dc.update(extract_attributes(node))

    weight_name = node.input[1] if len(node.input) > 1 else ""
    # Unnamed params are the historical [weight, bias] argument. Named ones are
    # the initializers actually wired to this node, so a constant bias next to
    # a computed weight must not be mistaken for that weight.
    param_names = {param.name for param in params if param.name}
    weight_is_constant = bool(params) and (
        not param_names or weight_name in param_names
    )
    if not weight_is_constant:
        # B is a graph or loop input. A functional Gemm consumes it at runtime.
        return DynamicGemm(
            transpose_activation=dc.get("transpose_activation", False),
            transpose_weight=dc.get("transpose_weight", True),
            weight_multiplier=dc.get("weight_multiplier", 1),
            bias_multiplier=dc.get("bias_multiplier", 1),
        )

    kwargs = {}
    weight, bias = extract_params(params)
    kwargs["bias"] = bias is not None
    kwargs["in_features"] = weight.dims[1]
    kwargs["out_features"] = weight.dims[0]

    # initialize layer and load weights
    layer = nn.Linear(**kwargs)
    load_params(layer, weight, bias)

    # apply onnx gemm attributes
    if dc.get("transpose_weight"):
        layer.weight.data = layer.weight.data.t()

    layer.weight.data *= dc.get("weight_multiplier")
    if layer.bias is not None:
        layer.bias.data *= dc.get("bias_multiplier")

    if dc.get("transpose_activation"):
        layer = nn.Sequential(Transpose(), layer)

    return layer


_DEFAULT_ACTIVATIONS = {
    "RNN": ["tanh"],
    "GRU": ["sigmoid", "tanh"],
    "LSTM": ["sigmoid", "tanh", "tanh"],
}


def _lowered(values):
    if values is None:
        return None
    return [value.lower() for value in values]


def _torch_activations(op_type, activations, bidirectional):
    """Whether nn.RNN/GRU/LSTM already implements this activation list.

    The comparison is case-insensitive. A bidirectional node may repeat the
    per-direction list. RNN's ReLU is the other nonlinearity torch implements.
    """
    if activations is None:
        return True
    lowered = _lowered(activations)
    defaults = _DEFAULT_ACTIVATIONS[op_type]
    if lowered == defaults:
        return True
    if bidirectional and lowered == defaults + defaults:
        return True
    if op_type == "RNN" and set(lowered) == {"relu"} and len(lowered) in (1, 2):
        if bidirectional or len(lowered) == 1:
            return True
    return False


def _needs_recurrent_fallback(op_type, dc, peephole=False):
    """Attributes that the torch recurrent modules cannot express."""
    if dc.get("clip") is not None:
        return True
    if dc.get("activation_alpha"):
        return True
    if dc.get("activation_beta"):
        return True
    if peephole:
        return True
    if op_type == "LSTM" and dc.get("input_forget"):
        return True
    return not _torch_activations(
        op_type, dc.get("activations"), dc.get("direction") == "bidirectional"
    )


def _recurrent_attributes(node, op_type, defaults):
    dc = dict(defaults)
    dc.update(extract_attributes(node))
    if dc["direction"] not in ("forward", "reverse", "bidirectional"):
        raise ValueError("{} direction {}.".format(op_type, dc["direction"]))
    if dc["hidden_size"] is None:
        raise ValueError("{} hidden_size is None.".format(op_type))
    if dc["layout"] not in (0, 1):
        raise NotImplementedError("{} layout={}.".format(op_type, dc["layout"]))
    return dc


def _runtime_input(node, weights, index):
    if index >= len(node.input):
        return False
    name = node.input[index]
    return bool(name) and name not in weights


def _check_lstm_state_inputs(node, weights):
    """Runtime initial_c cannot follow a constant or omitted initial_h.

    Standard layers receive only the inputs that are real activations, in
    order, so a constant initial_h would make a runtime initial_c arrive as h_0.
    """
    initial_h = _runtime_input(node, weights, 5)
    initial_c = _runtime_input(node, weights, 6)
    if initial_c and not initial_h:
        raise NotImplementedError(
            "LSTM initial_c is a runtime input but initial_h is not "
            "(initial_h={}, initial_c={}).".format(
                node.input[5] if len(node.input) > 5 else "",
                node.input[6],
            )
        )


def _build_fallback(
    op_type, dc, W, R, B, sequence_lens, initial_h, initial_c=None, peephole=None
):
    return OnnxRecurrent(
        op_type,
        W,
        R,
        B,
        peephole,
        hidden_size=dc["hidden_size"],
        direction=dc["direction"],
        activations=dc.get("activations"),
        activation_alpha=dc.get("activation_alpha"),
        activation_beta=dc.get("activation_beta"),
        clip=dc.get("clip"),
        layout=dc["layout"],
        linear_before_reset=dc.get("linear_before_reset", 0),
        input_forget=dc.get("input_forget", 0),
        sequence_lens=sequence_lens,
        initial_h=initial_h,
        initial_c=initial_c,
    )


def extract_sequence_lens(par_name, weights, op_type):
    if par_name not in weights:
        raise NotImplementedError(
            "{} sequence_lens has to be a constant.".format(op_type)
        )
    return torch.tensor(numpy_helper.to_array(weights[par_name]))


def check_rnn_inputs_are_constant(node, weights, names):
    """
    Reject recurrent inputs that do not come from an initializer.

    The weights are baked into a torch module at conversion time, so they cannot
    arrive at runtime. Only initial_h and initial_c can, and the wrapper's
    forward relies on that: it reads the remaining inputs by position.
    """
    for index, name in names.items():
        if index < len(node.input):
            par_name = node.input[index]
            if par_name != "" and par_name not in weights:
                raise NotImplementedError(
                    "{} {} has to be a constant.".format(node.op_type, name)
                )


def extract_and_load_params_gru(node, weights):
    """Extract and load parameters for GRU layer."""
    X = None
    W = None
    R = None
    B = None
    sequence_lens = None
    initial_h = None

    for par_ix, par_name in enumerate(node.input):
        if par_ix == 0:
            if par_name in weights:
                X = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 1:
            if par_name in weights:
                W = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 2:
            if par_name in weights:
                R = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 3:
            if par_name != "" and par_name in weights:
                B = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 4:
            if par_name != "":
                sequence_lens = extract_sequence_lens(par_name, weights, node.op_type)
        elif par_ix == 5:
            if par_name != "" and par_name in weights:
                initial_h = torch.tensor(numpy_helper.to_array(weights[par_name]))
    return (X, W, R, B, sequence_lens, initial_h)


def extract_and_load_params_lstm(node, weights):
    X = None
    W = None
    R = None
    B = None
    sequence_lens = None
    initial_h = None
    initial_c = None
    P = None

    for par_ix, par_name in enumerate(node.input):
        if par_ix == 0:
            if par_name in weights:
                X = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 1:
            if par_name in weights:
                W = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 2:
            if par_name in weights:
                R = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 3:
            if par_name != "" and par_name in weights:
                B = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 4:
            if par_name != "":
                sequence_lens = extract_sequence_lens(par_name, weights, node.op_type)
        elif par_ix == 5:
            if par_name != "" and par_name in weights:
                initial_h = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 6:
            if par_name != "" and par_name in weights:
                initial_c = torch.tensor(numpy_helper.to_array(weights[par_name]))
        elif par_ix == 7:
            if par_name != "" and par_name in weights:
                P = torch.tensor(numpy_helper.to_array(weights[par_name]))
    return (X, W, R, B, sequence_lens, initial_h, initial_c, P)


def convert_lstm_layer(node, weights):
    """Convert LSTM layer from onnx node and params."""
    check_rnn_inputs_are_constant(node, weights, {1: "W", 2: "R", 3: "B", 7: "P"})
    params_tuple = extract_and_load_params_lstm(node, weights)
    X, W, R, B, sequence_lens, initial_h, initial_c, P = params_tuple
    _check_lstm_state_inputs(node, weights)

    dc = _recurrent_attributes(
        node,
        "LSTM",
        dict(
            activation_alpha=None,
            activation_beta=None,
            activations=None,
            clip=None,
            direction="forward",
            hidden_size=None,
            input_forget=0,
            layout=0,
        ),
    )
    if _needs_recurrent_fallback("LSTM", dc, peephole=P is not None):
        return _build_fallback(
            "LSTM",
            dc,
            W,
            R,
            B,
            sequence_lens,
            initial_h,
            initial_c,
            P,
        )

    kwargs = {
        "input_size": W.shape[2],
        "hidden_size": dc["hidden_size"],
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0,
        "bidirectional": dc["direction"] == "bidirectional",
    }
    lstm_layer = nn.LSTM(**kwargs)

    input_size = kwargs["input_size"]
    hidden_size = kwargs["hidden_size"]
    num_directions = kwargs["bidirectional"] + 1
    num_layers = 1
    if B is None:
        # The ONNX bias is optional, torch's LSTM always has one
        B = torch.zeros(num_directions, 8 * hidden_size, dtype=W.dtype)
    if kwargs["bidirectional"]:
        # Set input-hidden weights
        W_iofc = W.transpose(0, 1).view(4 * hidden_size, num_directions, input_size)
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            W_ifco = torch.cat(
                tensors=(
                    W_iofc[0:hidden_size, dir_dim, :],
                    W_iofc[2 * hidden_size : 4 * hidden_size, dir_dim, :],
                    W_iofc[hidden_size : 2 * hidden_size, dir_dim, :],
                ),
                dim=0,
            )
            getattr(lstm_layer, "weight_ih_l0{}".format(dir_str)).data = W_ifco

        # Set hidden-hidden weights
        R_iofc = R.transpose(0, 1).view(4 * hidden_size, num_directions, hidden_size)
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            R_ifco = torch.cat(
                tensors=(
                    R_iofc[0:hidden_size, dir_dim, :],
                    R_iofc[2 * hidden_size : 4 * hidden_size, dir_dim, :],
                    R_iofc[hidden_size : 2 * hidden_size, dir_dim, :],
                ),
                dim=0,
            )
            getattr(lstm_layer, "weight_hh_l0{}".format(dir_str)).data = R_ifco

        # Set input-hidden biases
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            Wb_iofc = B[dir_dim, 0 : 4 * hidden_size]
            Wb_ifco = torch.cat(
                tensors=(
                    Wb_iofc[0:hidden_size],
                    Wb_iofc[2 * hidden_size : 4 * hidden_size],
                    Wb_iofc[hidden_size : 2 * hidden_size],
                ),
                dim=0,
            )
            getattr(lstm_layer, "bias_ih_l0{}".format(dir_str)).data = Wb_ifco

        # Set hidden-hidden biases
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            Rb_iofc = B[dir_dim, 4 * hidden_size :]
            Rb_ifco = torch.cat(
                tensors=(
                    Rb_iofc[0:hidden_size],
                    Rb_iofc[2 * hidden_size : 4 * hidden_size],
                    Rb_iofc[hidden_size : 2 * hidden_size],
                ),
                dim=0,
            )
            getattr(lstm_layer, "bias_hh_l0{}".format(dir_str)).data = Rb_ifco
    else:
        # Set input-hidden weights
        W_iofc = W.transpose(0, 1).view(4 * hidden_size, input_size)
        W_ifco = torch.cat(
            tensors=(
                W_iofc[0:hidden_size, :],
                W_iofc[2 * hidden_size : 4 * hidden_size, :],
                W_iofc[hidden_size : 2 * hidden_size, :],
            ),
            dim=0,
        )
        getattr(lstm_layer, "weight_ih_l0").data = W_ifco

        # Set hidden-hidden weights
        R_iofc = R.transpose(0, 1).view(4 * hidden_size, hidden_size)
        R_ifco = torch.cat(
            tensors=(
                R_iofc[0:hidden_size, :],
                R_iofc[2 * hidden_size : 4 * hidden_size, :],
                R_iofc[hidden_size : 2 * hidden_size, :],
            ),
            dim=0,
        )
        getattr(lstm_layer, "weight_hh_l0").data = R_ifco

        # Set input-hidden biases
        Wb_iofc = B[0, 0 : 4 * hidden_size]
        Wb_ifco = torch.cat(
            tensors=(
                Wb_iofc[0:hidden_size],
                Wb_iofc[2 * hidden_size : 4 * hidden_size],
                Wb_iofc[hidden_size : 2 * hidden_size],
            ),
            dim=0,
        )
        getattr(lstm_layer, "bias_ih_l0").data = Wb_ifco

        # Set hidden-hidden biases
        Rb_iofc = B[0, 4 * hidden_size :]
        Rb_ifco = torch.cat(
            tensors=(
                Rb_iofc[0:hidden_size],
                Rb_iofc[2 * hidden_size : 4 * hidden_size],
                Rb_iofc[hidden_size : 2 * hidden_size],
            ),
            dim=0,
        )
        getattr(lstm_layer, "bias_hh_l0").data = Rb_ifco

    layer = LSTMWrapper(
        lstm_layer,
        reverse=dc["direction"] == "reverse",
        sequence_lens=sequence_lens,
        layout=dc["layout"],
        initial_h=initial_h,
        initial_c=initial_c,
    )
    return layer


def convert_gru_layer(node, weights):
    """Convert GRU layer from onnx node and params."""
    check_rnn_inputs_are_constant(node, weights, {1: "W", 2: "R", 3: "B"})
    params_tuple = extract_and_load_params_gru(node, weights)
    X, W, R, B, sequence_lens, initial_h = params_tuple

    dc = _recurrent_attributes(
        node,
        "GRU",
        dict(
            activation_alpha=None,
            activation_beta=None,
            activations=None,
            clip=None,
            direction="forward",
            hidden_size=None,
            layout=0,
            linear_before_reset=0,  # ONNX spec default
        ),
    )
    if _needs_recurrent_fallback("GRU", dc):
        return _build_fallback("GRU", dc, W, R, B, sequence_lens, initial_h)
    # linear_before_reset is supported for both 0 and 1 on the torch GRU path

    kwargs = {
        "input_size": W.shape[2],
        "hidden_size": dc["hidden_size"],
        "num_layers": 1,
        "bias": True,
        "batch_first": False,
        "dropout": 0,
        "bidirectional": dc["direction"] == "bidirectional",
    }
    gru_layer = nn.GRU(**kwargs)

    input_size = kwargs["input_size"]
    hidden_size = kwargs["hidden_size"]
    num_directions = kwargs["bidirectional"] + 1
    num_layers = 1
    if B is None:
        # The ONNX bias is optional, torch's GRU always has one
        B = torch.zeros(num_directions, 6 * hidden_size, dtype=W.dtype)

    # ONNX GRU gate order: [z, r, h] (update, reset, hidden)
    # PyTorch GRU gate order: [r, z, n] (reset, input/update, new)
    # Need to reorder from ONNX to PyTorch

    if kwargs["bidirectional"]:
        # Set input-hidden weights
        W_zrh = W.transpose(0, 1).view(3 * hidden_size, num_directions, input_size)
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
            W_rzn = torch.cat(
                tensors=(
                    W_zrh[hidden_size : 2 * hidden_size, dir_dim, :],  # r
                    W_zrh[0:hidden_size, dir_dim, :],  # z
                    W_zrh[2 * hidden_size : 3 * hidden_size, dir_dim, :],  # n/h
                ),
                dim=0,
            )
            getattr(gru_layer, "weight_ih_l0{}".format(dir_str)).data = W_rzn

        # Set hidden-hidden weights
        R_zrh = R.transpose(0, 1).view(3 * hidden_size, num_directions, hidden_size)
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
            R_rzn = torch.cat(
                tensors=(
                    R_zrh[hidden_size : 2 * hidden_size, dir_dim, :],  # r
                    R_zrh[0:hidden_size, dir_dim, :],  # z
                    R_zrh[2 * hidden_size : 3 * hidden_size, dir_dim, :],  # n/h
                ),
                dim=0,
            )
            getattr(gru_layer, "weight_hh_l0{}".format(dir_str)).data = R_rzn

        # Set input-hidden biases
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            Wb_zrh = B[dir_dim, 0 : 3 * hidden_size]
            # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
            Wb_rzn = torch.cat(
                tensors=(
                    Wb_zrh[hidden_size : 2 * hidden_size],  # r
                    Wb_zrh[0:hidden_size],  # z
                    Wb_zrh[2 * hidden_size : 3 * hidden_size],  # n/h
                ),
                dim=0,
            )
            getattr(gru_layer, "bias_ih_l0{}".format(dir_str)).data = Wb_rzn

        # Set hidden-hidden biases
        for dir_dim, dir_str in [(0, ""), (1, "_reverse")]:
            Rb_zrh = B[dir_dim, 3 * hidden_size :]
            # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
            Rb_rzn = torch.cat(
                tensors=(
                    Rb_zrh[hidden_size : 2 * hidden_size],  # r
                    Rb_zrh[0:hidden_size],  # z
                    Rb_zrh[2 * hidden_size : 3 * hidden_size],  # n/h
                ),
                dim=0,
            )
            getattr(gru_layer, "bias_hh_l0{}".format(dir_str)).data = Rb_rzn
    else:
        # Set input-hidden weights
        W_zrh = W.transpose(0, 1).view(3 * hidden_size, input_size)
        # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
        W_rzn = torch.cat(
            tensors=(
                W_zrh[hidden_size : 2 * hidden_size, :],  # r
                W_zrh[0:hidden_size, :],  # z
                W_zrh[2 * hidden_size : 3 * hidden_size, :],  # n/h
            ),
            dim=0,
        )
        getattr(gru_layer, "weight_ih_l0").data = W_rzn

        # Set hidden-hidden weights
        R_zrh = R.transpose(0, 1).view(3 * hidden_size, hidden_size)
        # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
        R_rzn = torch.cat(
            tensors=(
                R_zrh[hidden_size : 2 * hidden_size, :],  # r
                R_zrh[0:hidden_size, :],  # z
                R_zrh[2 * hidden_size : 3 * hidden_size, :],  # n/h
            ),
            dim=0,
        )
        getattr(gru_layer, "weight_hh_l0").data = R_rzn

        # Set input-hidden biases
        Wb_zrh = B[0, 0 : 3 * hidden_size]
        # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
        Wb_rzn = torch.cat(
            tensors=(
                Wb_zrh[hidden_size : 2 * hidden_size],  # r
                Wb_zrh[0:hidden_size],  # z
                Wb_zrh[2 * hidden_size : 3 * hidden_size],  # n/h
            ),
            dim=0,
        )
        getattr(gru_layer, "bias_ih_l0").data = Wb_rzn

        # Set hidden-hidden biases
        Rb_zrh = B[0, 3 * hidden_size :]
        # Reorder from ONNX [z, r, h] to PyTorch [r, z, n]
        Rb_rzn = torch.cat(
            tensors=(
                Rb_zrh[hidden_size : 2 * hidden_size],  # r
                Rb_zrh[0:hidden_size],  # z
                Rb_zrh[2 * hidden_size : 3 * hidden_size],  # n/h
            ),
            dim=0,
        )
        getattr(gru_layer, "bias_hh_l0").data = Rb_rzn

    layer = GRUWrapper(
        gru_layer,
        linear_before_reset=dc["linear_before_reset"],
        reverse=dc["direction"] == "reverse",
        sequence_lens=sequence_lens,
        layout=dc["layout"],
        initial_h=initial_h,
    )
    return layer


def convert_rnn_layer(node, weights):
    """Convert RNN layer from onnx node and params."""
    # ONNX RNN has the same input layout as GRU
    check_rnn_inputs_are_constant(node, weights, {1: "W", 2: "R", 3: "B"})
    params_tuple = extract_and_load_params_gru(node, weights)
    X, W, R, B, sequence_lens, initial_h = params_tuple

    dc = _recurrent_attributes(
        node,
        "RNN",
        dict(
            activation_alpha=None,
            activation_beta=None,
            activations=None,
            clip=None,
            direction="forward",
            hidden_size=None,
            layout=0,
        ),
    )
    if _needs_recurrent_fallback("RNN", dc):
        return _build_fallback("RNN", dc, W, R, B, sequence_lens, initial_h)

    nonlinearity = "tanh"
    if dc["activations"] is not None and set(a.lower() for a in dc["activations"]) == {
        "relu"
    }:
        nonlinearity = "relu"

    kwargs = {
        "input_size": W.shape[2],
        "hidden_size": dc["hidden_size"],
        "num_layers": 1,
        "nonlinearity": nonlinearity,
        "bias": True,
        "batch_first": False,
        "dropout": 0,
        "bidirectional": dc["direction"] == "bidirectional",
    }
    rnn_layer = nn.RNN(**kwargs)

    hidden_size = kwargs["hidden_size"]
    directions = [(0, "")]
    if kwargs["bidirectional"]:
        directions.append((1, "_reverse"))

    for dir_dim, dir_str in directions:
        getattr(rnn_layer, "weight_ih_l0{}".format(dir_str)).data = W[dir_dim]
        getattr(rnn_layer, "weight_hh_l0{}".format(dir_str)).data = R[dir_dim]
        if B is None:
            Wb = torch.zeros(hidden_size, dtype=W.dtype)
            Rb = torch.zeros(hidden_size, dtype=W.dtype)
        else:
            Wb = B[dir_dim, :hidden_size]
            Rb = B[dir_dim, hidden_size:]
        getattr(rnn_layer, "bias_ih_l0{}".format(dir_str)).data = Wb
        getattr(rnn_layer, "bias_hh_l0{}".format(dir_str)).data = Rb

    layer = RNNWrapper(
        rnn_layer,
        reverse=dc["direction"] == "reverse",
        sequence_lens=sequence_lens,
        layout=dc["layout"],
        initial_h=initial_h,
    )
    return layer
