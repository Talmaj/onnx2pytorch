import warnings

import onnx
from onnx import numpy_helper

from onnx2pytorch.utils import (
    extract_padding_params_for_conv_layer,
    extract_padding_params,
)

TENSOR_PROTO_MAPPING = dict([i[::-1] for i in onnx.TensorProto.DataType.items()])

AttributeType = dict(
    UNDEFINED=0,
    FLOAT=1,
    INT=2,
    STRING=3,
    TENSOR=4,
    GRAPH=5,
    FLOATS=6,
    INTS=7,
    STRINGS=8,
    TENSORS=9,
    GRAPHS=10,
    SPARSE_TENSOR=11,
    SPARSE_TENSORS=12,
    TYPE_PROTO=13,
    TYPE_PROTOS=14,
)


def extract_attr_values(attr):
    """Extract onnx attribute values."""
    if attr.type == AttributeType["INT"]:
        value = attr.i
    elif attr.type == AttributeType["FLOAT"]:
        value = attr.f
    elif attr.type == AttributeType["INTS"]:
        value = tuple(attr.ints)
    elif attr.type == AttributeType["FLOATS"]:
        value = tuple(attr.floats)
    elif attr.type == AttributeType["TENSOR"]:
        value = numpy_helper.to_array(attr.t)
    elif attr.type == AttributeType["STRING"]:
        value = attr.s.decode()
    elif attr.type == AttributeType["GRAPH"]:
        value = attr.g
    elif attr.type == AttributeType["STRINGS"]:
        value = tuple(s.decode() for s in attr.strings)
    elif attr.type == AttributeType["TENSORS"]:
        value = tuple(numpy_helper.to_array(t) for t in attr.tensors)
    elif attr.type == AttributeType["GRAPHS"]:
        value = tuple(attr.graphs)
    elif attr.type == AttributeType["SPARSE_TENSOR"]:
        value = attr.sparse_tensor
    elif attr.type == AttributeType["SPARSE_TENSORS"]:
        value = tuple(attr.sparse_tensors)
    elif attr.type == AttributeType["TYPE_PROTO"]:
        value = attr.tp
    elif attr.type == AttributeType["TYPE_PROTOS"]:
        value = tuple(attr.type_protos)
    else:
        raise NotImplementedError(
            "Extraction of attribute type {} not implemented.".format(attr.type)
        )
    return value


def extract_attributes(node):
    """Extract onnx attributes. Map onnx feature naming to pytorch."""
    kwargs = {}
    for attr in node.attribute:
        if attr.name == "activation_alpha":
            kwargs["activation_alpha"] = extract_attr_values(attr)
        elif attr.name == "activation_beta":
            kwargs["activation_beta"] = extract_attr_values(attr)
        elif attr.name == "activations":
            kwargs["activations"] = extract_attr_values(attr)
        elif attr.name == "alpha":
            if node.op_type == "LeakyRelu":
                kwargs["negative_slope"] = extract_attr_values(attr)
            elif node.op_type in ("Elu", "ThresholdedRelu"):
                kwargs["alpha"] = extract_attr_values(attr)
            elif node.op_type == "HardSigmoid":
                kwargs["alpha"] = extract_attr_values(attr)
            else:
                kwargs["weight_multiplier"] = extract_attr_values(attr)
        elif attr.name == "auto_pad":
            value = extract_attr_values(attr)
            if value == "NOTSET":
                # NOTSET means explicit padding is used, don't add to kwargs
                pass
            elif value in ("VALID", "SAME_UPPER", "SAME_LOWER"):
                kwargs["auto_pad"] = value
            else:
                raise NotImplementedError(
                    "auto_pad={} functionality not implemented.".format(value)
                )
        elif attr.name == "axis" and node.op_type == "Flatten":
            kwargs["start_dim"] = extract_attr_values(attr)
        elif attr.name == "axis" and node.op_type == "LayerNormalization":
            kwargs["axis"] = extract_attr_values(attr)
        elif attr.name == "axis" or attr.name == "axes":
            v = extract_attr_values(attr)
            if isinstance(v, (tuple, list)) and len(v) == 1:
                kwargs["dim"] = v[0]
            else:
                kwargs["dim"] = v
        elif attr.name == "beta":
            if node.op_type == "HardSigmoid":
                kwargs["beta"] = extract_attr_values(attr)
            else:
                kwargs["bias_multiplier"] = extract_attr_values(attr)
        elif attr.name == "body":
            kwargs["body"] = extract_attr_values(attr)
        elif attr.name == "ceil_mode":
            kwargs["ceil_mode"] = bool(extract_attr_values(attr))
        elif attr.name == "center_point_box":
            kwargs["center_point_box"] = extract_attr_values(attr)
        elif attr.name == "clip":
            kwargs["clip"] = extract_attr_values(attr)
        elif attr.name == "coordinate_transformation_mode":
            arg = extract_attr_values(attr)
            if arg == "align_corners":
                kwargs["align_corners"] = True
            else:
                warnings.warn(
                    "Pytorch's interpolate uses no coordinate_transformation_mode={}. "
                    "Result might differ.".format(arg)
                )
        elif attr.name == "dilations":
            kwargs["dilation"] = extract_attr_values(attr)
        elif attr.name == "direction":
            kwargs["direction"] = extract_attr_values(attr)
        elif attr.name == "dtype":
            kwargs["dtype"] = extract_attr_values(attr)
        elif attr.name == "else_branch":
            kwargs["else_branch"] = extract_attr_values(attr)
        elif attr.name == "ends":
            kwargs["ends"] = extract_attr_values(attr)
        elif attr.name == "epsilon":
            kwargs["eps"] = extract_attr_values(attr)
        elif attr.name == "group":
            kwargs["groups"] = extract_attr_values(attr)
        elif attr.name == "hidden_size":
            kwargs["hidden_size"] = extract_attr_values(attr)
        elif attr.name == "high":
            kwargs["high"] = extract_attr_values(attr)
        elif attr.name == "input_forget":
            kwargs["input_forget"] = extract_attr_values(attr)
        elif attr.name == "keepdims":
            kwargs["keepdim"] = bool(extract_attr_values(attr))
        elif attr.name == "kernel_shape":
            kwargs["kernel_size"] = extract_attr_values(attr)
        elif attr.name == "largest":
            kwargs["largest"] = extract_attr_values(attr)
        elif attr.name == "layout":
            kwargs["layout"] = extract_attr_values(attr)
        elif attr.name == "linear_before_reset":
            kwargs["linear_before_reset"] = extract_attr_values(attr)
        elif attr.name == "low":
            kwargs["low"] = extract_attr_values(attr)
        elif attr.name == "max":
            kwargs["max"] = extract_attr_values(attr)
        elif attr.name == "min":
            kwargs["min"] = extract_attr_values(attr)
        elif attr.name == "mode":
            kwargs["mode"] = extract_attr_values(attr)
        elif attr.name == "momentum":
            kwargs["momentum"] = extract_attr_values(attr)
        elif attr.name == "noop_with_empty_axes":
            kwargs["noop_with_empty_axes"] = extract_attr_values(attr)
        elif attr.name == "output_shape" and node.op_type == "ConvTranspose":
            raise NotImplementedError(
                "ConvTranspose with dynamic padding not implemented."
            )
        elif attr.name == "pads":
            params = extract_attr_values(attr)
            if node.op_type == "Pad":
                kwargs["padding"] = extract_padding_params(params)
            else:
                # Works for Conv, MaxPooling and other layers from convert_layer func
                kwargs["padding"] = extract_padding_params_for_conv_layer(params)
        elif attr.name == "perm":
            kwargs["dims"] = extract_attr_values(attr)
        elif attr.name == "repeats":
            kwargs["repeats"] = extract_attr_values(attr)
        elif attr.name == "seed":
            kwargs["seed"] = extract_attr_values(attr)
        elif attr.name == "sorted":
            kwargs["sorted"] = extract_attr_values(attr)
        elif attr.name == "sparse_value":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "spatial":
            kwargs["spatial"] = extract_attr_values(attr)  # Batch norm parameter
        elif attr.name == "split":
            kwargs["split_size_or_sections"] = extract_attr_values(attr)
        elif attr.name == "strides":
            kwargs["stride"] = extract_attr_values(attr)
        elif attr.name == "starts":
            kwargs["starts"] = extract_attr_values(attr)
        elif attr.name == "then_branch":
            kwargs["then_branch"] = extract_attr_values(attr)
        elif attr.name == "to":
            kwargs["dtype"] = TENSOR_PROTO_MAPPING[extract_attr_values(attr)].lower()
        elif attr.name == "type":
            kwargs["type"] = extract_attr_values(attr)
        elif attr.name == "transB":
            kwargs["transpose_weight"] = not extract_attr_values(attr)
        elif attr.name == "transA":
            kwargs["transpose_activation"] = bool(extract_attr_values(attr))
        elif attr.name == "value":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "value_float":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "value_floats":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "value_int":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "value_ints":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "value_string":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "value_strings":
            kwargs["constant"] = extract_attr_values(attr)
        elif attr.name == "output_padding":
            kwargs["output_padding"] = extract_attr_values(attr)
        elif node.op_type == "Resize":
            # These parameters are not used, warn in Resize operator
            kwargs[attr.name] = extract_attr_values(attr)
        else:
            raise NotImplementedError(
                "Extraction of attribute {} not implemented.".format(attr.name)
            )
    return kwargs
