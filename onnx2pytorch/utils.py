import io

import torch
import numpy as np
import onnx

try:
    import onnxruntime as ort
except ImportError:
    ort = None


def value_wrapper(value):
    def callback(*args, **kwargs):
        return value

    return callback


def is_constant(value):
    return value.ndim == 0 or value.shape == torch.Size([1])


def is_symmetric(params):
    """
    Check if parameters are symmetric, all values [2,2,2,2].
    Then we can use only [2,2].
    """
    assert len(params) // 2 == len(params) / 2, "Non even numer of parameters."
    idx = len(params) // 2
    for i in range(0, idx):
        assert params[i] == params[idx + i], "Both sides should be the same."
    return True


def to_pytorch_params(params):
    """
    Padding in onnx is different than in pytorch. That is why we need to
    check if they are symmetric and cut half.
    """
    if is_symmetric(params):
        return params[: len(params) // 2]
    else:
        raise ValueError("Parameters need to be symmetric to work with pytorch.")


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

    Returns
    -------
    value: list[np.ndarray]
        Value of the activation with activation_name.
    """
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"

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


def get_inputs_sample(onnx_model, to_torch=False):
    """Get inputs sample from onnx model."""
    assert ort is not None, "onnxruntime needed. pip install onnxruntime"

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    inputs = sess.get_inputs()
    input_names = [x.name for x in inputs]
    input_tensors = [
        np.abs(np.random.rand(*get_shape(x)).astype(get_type(x))) for x in inputs
    ]
    if to_torch:
        input_tensors = [torch.from_numpy(x) for x in input_tensors]
    return dict(zip(input_names, input_tensors))
