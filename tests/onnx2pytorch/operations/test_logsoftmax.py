import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


@pytest.mark.parametrize(
    "axis,input_shape",
    [
        (-1, [2, 3, 4]),  # Default axis=-1
        (0, [2, 3, 4]),
        (1, [2, 3, 4]),
        (2, [2, 3, 4]),
        (-2, [2, 3, 4]),
        (1, [5, 10]),  # 2D input
        (-1, [8]),  # 1D input
    ],
)
def test_logsoftmax_onnxruntime(axis, input_shape):
    """Test LogSoftmax against onnxruntime."""
    np.random.seed(42)

    # Create input
    X = np.random.randn(*input_shape).astype(np.float32)

    # Create ONNX graph with LogSoftmax node
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)

    logsoftmax_node = helper.make_node(
        "LogSoftmax",
        inputs=["X"],
        outputs=["Y"],
        axis=axis,
    )

    graph = helper.make_graph(
        [logsoftmax_node],
        "logsoftmax_test",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )

    # Run with onnxruntime
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_outputs = ort_session.run(None, {"X": X})
    expected_Y = ort_outputs[0]

    # Convert to PyTorch and run
    o2p_model = ConvertModel(model, experimental=True)
    X_torch = torch.from_numpy(X)

    with torch.no_grad():
        o2p_output = o2p_model(X_torch)

    # Compare outputs
    torch.testing.assert_close(
        o2p_output,
        torch.from_numpy(expected_Y),
        rtol=1e-5,
        atol=1e-5,
    )


def test_logsoftmax_default_axis():
    """Test LogSoftmax with default axis=-1."""
    np.random.seed(42)

    input_shape = [2, 3, 4]
    X = np.random.randn(*input_shape).astype(np.float32)

    # Create ONNX graph WITHOUT specifying axis (should default to -1)
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)

    logsoftmax_node = helper.make_node(
        "LogSoftmax",
        inputs=["X"],
        outputs=["Y"],
        # No axis specified - should default to -1
    )

    graph = helper.make_graph(
        [logsoftmax_node],
        "logsoftmax_test",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)], ir_version=8
    )

    # Run with onnxruntime
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_outputs = ort_session.run(None, {"X": X})
    expected_Y = ort_outputs[0]

    # Convert to PyTorch and run
    o2p_model = ConvertModel(model, experimental=True)
    X_torch = torch.from_numpy(X)

    with torch.no_grad():
        o2p_output = o2p_model(X_torch)

    # Compare outputs
    torch.testing.assert_close(
        o2p_output,
        torch.from_numpy(expected_Y),
        rtol=1e-5,
        atol=1e-5,
    )


def test_logsoftmax_properties():
    """Test mathematical properties of LogSoftmax."""
    # LogSoftmax(x) = log(Softmax(x))
    X = torch.randn(2, 5)

    logsoftmax_output = torch.nn.functional.log_softmax(X, dim=-1)
    softmax_output = torch.nn.functional.softmax(X, dim=-1)
    log_of_softmax = torch.log(softmax_output)

    torch.testing.assert_close(logsoftmax_output, log_of_softmax, rtol=1e-5, atol=1e-5)

    # Sum of exp(log_softmax) should be 1
    sum_exp = torch.exp(logsoftmax_output).sum(dim=-1)
    torch.testing.assert_close(sum_exp, torch.ones_like(sum_exp), rtol=1e-5, atol=1e-5)
