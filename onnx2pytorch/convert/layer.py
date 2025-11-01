import numpy as np
import torch
from torch import nn
from onnx import numpy_helper

from onnx2pytorch.operations import (
    AutoPad,
    BatchNormWrapper,
    GRUWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
)
from onnx2pytorch.convert.attribute import extract_attributes, extract_attr_values


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


def convert_layer(node, layer_type, params=None):
    """Use to convert Conv, MaxPool, AvgPool layers."""
    assert layer_type in [
        "Conv",
        "ConvTranspose",
        "MaxPool",
        "AvgPool",
    ], "Incorrect layer type: {}".format(layer_type)
    kwargs = extract_attributes(node)
    kernel_size_length = len(kwargs["kernel_size"])
    try:
        layer = getattr(nn, "{}{}d".format(layer_type, kernel_size_length))
    except AttributeError:
        raise ValueError(
            "Unexpected length of kernel_size dimension: {}".format(kernel_size_length)
        )

    # Handle auto_pad attribute
    pad_layer = None
    auto_pad = kwargs.pop("auto_pad", "NOTSET")
    if auto_pad == "VALID":
        # No padding
        kwargs["padding"] = 0
    elif auto_pad in ("SAME_UPPER", "SAME_LOWER"):
        # Create dynamic padding layer
        # Note: explicit pads are ignored when auto_pad is not NOTSET (per ONNX spec)
        pad_layer = AutoPad(
            kernel_size=kwargs["kernel_size"],
            stride=kwargs.get("stride", 1),
            dilation=kwargs.get("dilation", 1),
            mode=auto_pad,
        )
        kwargs["padding"] = 0  # Conv layer itself should not pad

    if params:
        weight, bias = extract_params(params)
        kwargs["bias"] = bias is not None
        kwargs["in_channels"] = weight.dims[1] * kwargs.get("groups", 1)
        kwargs["out_channels"] = weight.dims[0]

        if layer_type == "ConvTranspose":
            kwargs["in_channels"], kwargs["out_channels"] = (
                kwargs["out_channels"],
                kwargs["in_channels"],
            )

        # if padding is a layer, remove from kwargs and prepend later
        if "padding" in kwargs and isinstance(kwargs["padding"], nn.Module):
            pad_layer = kwargs.pop("padding")

        # initialize layer and load weights
        layer = layer(**kwargs)
        load_params(layer, weight, bias)
    else:
        # initialize operations without parameters (MaxPool, AvgPool, etc.)
        if layer_type == "MaxPool":
            kwargs["return_indices"] = True

        # if padding is a layer, remove from kwargs and prepend later
        if "padding" in kwargs and isinstance(kwargs["padding"], nn.Module):
            pad_layer = kwargs.pop("padding")
        layer = layer(**kwargs)

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
    for attr in node.attribute:
        if attr.name in ["transA"] and extract_attr_values(attr) != 0:
            raise NotImplementedError(
                "Not implemented for attr.name={} and value!=0.".format(attr.name)
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

    return layer


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
            if par_name != "" and par_name in weights:
                sequence_lens = torch.tensor(numpy_helper.to_array(weights[par_name]))
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
            if par_name != "" and par_name in weights:
                sequence_lens = torch.tensor(numpy_helper.to_array(weights[par_name]))
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
    params_tuple = extract_and_load_params_lstm(node, weights)
    (X, W, R, B, sequence_lens, initial_h, initial_c, P) = params_tuple
    if initial_h is not None:
        raise NotImplementedError("LSTM initial_h not yet implemented.")
    if initial_c is not None:
        raise NotImplementedError("LSTM initial_c not yet implemented.")
    if P is not None:
        raise NotImplementedError("LSTM P not yet implemented.")

    dc = dict(
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction="forward",
        hidden_size=None,
        input_forget=0,
        layout=0,
    )
    dc.update(extract_attributes(node))
    if dc["activation_alpha"] is not None:
        raise NotImplementedError(
            "LSTM activation_alpha {}.".format(dc["activation_alpha"])
        )
    if dc["activation_beta"] is not None:
        raise NotImplementedError(
            "LSTM activation_beta {}.".format(dc["activation_beta"])
        )
    if dc["activations"] is not None:
        # TODO allow if torch-compatible activations are set explicitly
        raise NotImplementedError("LSTM activations {}.".format(dc["activations"]))
    if dc["clip"] is not None:
        raise NotImplementedError("LSTM clip {}".format(dc["clip"]))
    if dc["direction"] not in ("forward", "bidirectional"):
        raise ValueError("LSTM direction {}.".format(dc["direction"]))
    if dc["hidden_size"] is None:
        raise ValueError("LSTM hidden_size is None.")
    if dc["input_forget"] != 0:
        raise NotImplementedError("LSTM input_forget {}.".format(dc["input_forget"]))
    if dc["layout"] != 0:
        raise NotImplementedError(
            "LSTM not implemented for layout={}".format(dc["layout"])
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

    layer = LSTMWrapper(lstm_layer)
    return layer


def convert_gru_layer(node, weights):
    """Convert GRU layer from onnx node and params."""
    params_tuple = extract_and_load_params_gru(node, weights)
    (X, W, R, B, sequence_lens, initial_h) = params_tuple
    if initial_h is not None:
        raise NotImplementedError("GRU initial_h not yet implemented.")

    dc = dict(
        activation_alpha=None,
        activation_beta=None,
        activations=None,
        clip=None,
        direction="forward",
        hidden_size=None,
        layout=0,
        linear_before_reset=0,  # ONNX spec default
    )
    dc.update(extract_attributes(node))
    if dc["activation_alpha"] is not None:
        raise NotImplementedError(
            "GRU activation_alpha {}.".format(dc["activation_alpha"])
        )
    if dc["activation_beta"] is not None:
        raise NotImplementedError(
            "GRU activation_beta {}.".format(dc["activation_beta"])
        )
    if dc["activations"] is not None:
        # TODO allow if torch-compatible activations are set explicitly
        raise NotImplementedError("GRU activations {}.".format(dc["activations"]))
    if dc["clip"] is not None:
        raise NotImplementedError("GRU clip {}".format(dc["clip"]))
    if dc["direction"] not in ("forward", "bidirectional"):
        raise ValueError("GRU direction {}.".format(dc["direction"]))
    if dc["hidden_size"] is None:
        raise ValueError("GRU hidden_size is None.")
    if dc["layout"] != 0:
        raise NotImplementedError(
            "GRU not implemented for layout={}".format(dc["layout"])
        )
    # linear_before_reset is now supported for both 0 and 1

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

    layer = GRUWrapper(gru_layer, linear_before_reset=dc["linear_before_reset"])
    return layer
