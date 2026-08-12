import io
import math

import torch
import numpy as np
import onnx

from onnx2pytorch.dtypes import ONNX_DTYPE_TO_TORCH

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def value_wrapper(value):
    def callback(*args, **kwargs):
        return value

    return callback


class _OmittedInput:
    def __repr__(self):
        return "OMITTED_INPUT"


# ONNX marks an omitted optional node input with an empty input name.
OMITTED_INPUT = _OmittedInput()


def resolve_omitted_inputs(in_activations):
    """
    Drop omitted optional inputs at the end and pass on the remaining ones as None.

    Omitted inputs cannot simply be removed, as that would shift the inputs that
    follow them into the wrong positional argument of the operation.
    """
    while in_activations and in_activations[-1] is OMITTED_INPUT:
        in_activations.pop()
    return [None if act is OMITTED_INPUT else act for act in in_activations]


def cosine_window(size, periodic, coefficients):
    """
    Evaluate a cosine-sum window as defined by the ONNX windowing operators.

    Unlike torch.hann_window and friends there is no special case for size 1.
    """
    size = int(size)
    n = torch.arange(size, dtype=torch.float64)
    angle = 2 * math.pi * n / (size if periodic else size - 1)
    window = torch.zeros_like(n)
    for k, coefficient in enumerate(coefficients):
        window = window + coefficient * torch.cos(k * angle)
    return window


def get_random_generator(seed, device=None):
    """
    Build a generator seeded with the seed attribute of the ONNX random operators.

    Returns None if no seed is given, which makes torch use its global generator.
    """
    if seed is None:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    return generator


def get_torch_dtype(onnx_dtype):
    """Map an ONNX TensorProto data type to the corresponding torch dtype."""
    dtype = ONNX_DTYPE_TO_TORCH.get(onnx_dtype)
    if dtype is None:
        raise NotImplementedError(
            "ONNX dtype {} is not supported in PyTorch.".format(onnx_dtype)
        )
    return dtype


def is_constant(value):
    return value.ndim == 0 or value.shape == torch.Size([1])


def is_symmetric(params):
    """
    Check if parameters are symmetric, all values [2,2,2,2].
    Then we can use only [2,2].
    """
    assert len(params) // 2 == len(params) / 2, "Non even number of parameters."
    idx = len(params) // 2
    for i in range(0, idx):
        if params[i] != params[idx + i]:
            return False
    return True


def convert_onnx_pads_to_torch(pads):
    """
    Convert pads from ONNX to PyTorch convention.

    ONNX groups all dimension begins first, then all ends, in dimension order:
        [begin_d0, ..., begin_dN, end_d0, ..., end_dN]
    PyTorch interleaves begin and end per dimension, last dimension first:
        [begin_dN, end_dN, ..., begin_d0, end_d0]
    """
    pad_dim = len(pads) // 2
    if pad_dim == 0:
        return []
    begins = pads[:pad_dim]
    ends = pads[pad_dim:]
    torch_pads = []
    for i in range(pad_dim - 1, -1, -1):
        torch_pads.extend([begins[i], ends[i]])
    return torch_pads


def as_tuple(value, length):
    """Broadcast a scalar layer parameter to one value per spatial dimension."""
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,) * length


def extract_padding_params(params):
    """Extract padding parameters for Pad layers."""
    pads = convert_onnx_pads_to_torch(params)

    # Some padding modes do not support padding in batch and channel dimension.
    # In torch convention those are the trailing four values.
    if len(pads) > 4 and not any(pads[-4:]):
        pads = pads[:-4]
    return pads


def lowest_value(dtype):
    """The most negative value a dtype holds, the neutral pad for max pooling."""
    if dtype.is_floating_point:
        return float("-inf")
    return torch.iinfo(dtype).min


def extract_padding_params_for_conv_layer(params, value=0):
    """
    Padding params in onnx are different than in pytorch. That is why we need to
    check if they are symmetric and cut half or return a padding layer.

    Asymmetric pads have to be materialised, so they need a fill value. Pooling
    over the maximum has to ignore the pads instead of counting them as zeros.
    """
    if is_symmetric(params):
        return params[: len(params) // 2]
    # Conv pads cover spatial dimensions only, so nothing may be discarded.
    torch_pads = convert_onnx_pads_to_torch(params)
    if value == float("-inf"):
        from onnx2pytorch.operations.lowestpad import LowestPad

        return LowestPad(torch_pads)
    pad_layer = getattr(torch.nn, "ConstantPad{}d".format(len(params) // 2))
    return pad_layer(torch_pads, value=value)


def get_reduce_dims(data, dim, axes=None, noop_with_empty_axes=False):
    """
    Resolve the dimensions to reduce over.

    Axes are an attribute in older opset versions (dim) and an optional
    input in newer ones (axes). Returns None if reduction is a no-op.
    """
    if torch.is_tensor(axes) and axes.numel() == 0:
        axes = None
    dims = dim if axes is None else axes
    if dims is None:
        return None if noop_with_empty_axes else tuple(range(data.ndim))
    if isinstance(dims, int):
        return dims
    return tuple(int(d) for d in torch.atleast_1d(torch.as_tensor(dims)))


def as_input_dtype(result, data):
    """
    Cast a reduction back to the type of its input.

    ONNX reductions return the input type, while torch accumulates integers into
    int64 and takes roots and means in floating point. An integer result is
    truncated toward zero, which is what both onnx runtimes do.
    """
    if result.dtype == data.dtype:
        return result
    if result.dtype.is_floating_point and not data.dtype.is_floating_point:
        result = torch.trunc(result)
    return result.to(data.dtype)


def get_selection(indices, dim):
    """
    Give selection to assign values to specific indices at given dimension.
    Enables dimension to be dynamic:
        tensor[get_selection(indices, dim=2)] = values
    Alternatively the dimension is fixed in code syntax:
        tensor[:, :, indices] = values
    """
    assert dim >= 0, "Negative dimension not supported."
    # Behaviour with python lists is unfortunately not working the same.
    if isinstance(indices, list):
        indices = torch.tensor(indices)
    assert isinstance(indices, (torch.Tensor, np.ndarray))
    selection = [slice(None) for _ in range(dim + 1)]
    selection[dim] = indices
    return selection


def assign_values_to_dim(tensor, values, indices, dim, inplace=True):
    """
    Inplace tensor operation that assigns values to corresponding indices
    at given dimension.
    """
    if dim < 0:
        dim = dim + len(tensor.shape)
    selection = get_selection(indices, dim)
    if not inplace:
        tensor = tensor.clone()
    tensor[selection] = values
    return tensor


def get_type(x):
    """
    Extract type from onnxruntime input.

    Parameters
    ----------
    x: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg
    """
    if x.type.startswith("tensor"):
        typ = x.type[7:-1]
    else:
        raise NotImplementedError("For type: {}".format(x.type))

    if typ == "float":
        typ = "float32"
    elif typ == "double":
        typ = "float64"
    return typ


def get_shape(x, unknown_dim_size=1):
    """
    Extract shape from onnxruntime input.
    Replace unknown dimension by default with 1.

    Parameters
    ----------
    x: onnxruntime.capi.onnxruntime_pybind11_state.NodeArg
    unknown_dim_size: int
        Default: 1
    """
    shape = x.shape
    # replace unknown dimensions by default with 1
    shape = [i if isinstance(i, int) else unknown_dim_size for i in shape]
    return shape


def get_activation_value(onnx_model, inputs, activation_names):
    """
    Get activation value from an onnx model.

    Parameters
    ----------
    onnx_model: onnx.ModelProto
    inputs: list[np.ndarray]
    activation_names: list[str]
        Can be retrieved from onnx node: list(node.output)

    Returns
    -------
    value: list[np.ndarray]
        Value of the activation with activation_name.
    """
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"
    assert all(isinstance(x, np.ndarray) for x in inputs)

    if not isinstance(activation_names, (list, tuple)):
        activation_names = [activation_names]

    # clear output
    while len(onnx_model.graph.output):
        onnx_model.graph.output.pop()

    for activation_name in activation_names:
        activation_value = onnx.helper.ValueInfoProto()
        activation_value.name = activation_name
        onnx_model.graph.output.append(activation_value)

    buffer = io.BytesIO()
    onnx.save(onnx_model, buffer)
    buffer.seek(0)
    onnx_model_new = onnx.load(buffer)
    sess = ort.InferenceSession(onnx_model_new.SerializeToString())

    input_names = [x.name for x in sess.get_inputs()]
    if not isinstance(inputs, list):
        inputs = [inputs]
    inputs = dict(zip(input_names, inputs))

    return sess.run(None, inputs)


def get_inputs_names(onnx_graph):
    param_names = set([x.name for x in onnx_graph.initializer])
    input_names = [x.name for x in onnx_graph.input]
    input_names = [x for x in input_names if x not in param_names]
    return input_names


def get_inputs_sample(onnx_model, to_torch=False):
    """Get inputs sample from onnx model."""
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = sess.get_inputs()
    input_names = get_inputs_names(onnx_model.graph)
    input_tensors = [
        np.abs(np.random.rand(*get_shape(x)).astype(get_type(x))) for x in inputs
    ]
    if to_torch:
        input_tensors = [torch.tensor(x) for x in input_tensors]
    return dict(zip(input_names, input_tensors))


def get_outputs_names(onnx_graph):
    output_names = [x.name for x in onnx_graph.output]
    return output_names


def get_ops_names(onnx_graph):
    ops_used = set(node.op_type for node in onnx_graph.node)
    for node in onnx_graph.node:
        if node.op_type in ("Loop", "Scan", "SequenceMap"):
            for attr in node.attribute:
                if attr.name == "body":
                    ops_used |= get_ops_names(attr.g)
        elif node.op_type == "If":
            for attr in node.attribute:
                if attr.name == "then_branch":
                    ops_used |= get_ops_names(attr.g)
                elif attr.name == "else_branch":
                    ops_used |= get_ops_names(attr.g)
    return ops_used
