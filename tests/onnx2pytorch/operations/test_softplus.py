import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


@pytest.mark.parametrize(
    "input_shape",
    [
        [2, 3, 4],
        [5, 10],
        [8],
        [1, 1, 5, 5],
    ],
)
def test_softplus_default_onnxruntime(input_shape):
    """Test Softplus with default parameters against onnxruntime."""
    np.random.seed(42)

    # Create input
    X = np.random.randn(*input_shape).astype(np.float32)

    # Create ONNX graph with Softplus node (default parameters)
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)

    softplus_node = helper.make_node(
        "Softplus",
        inputs=["X"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [softplus_node],
        "softplus_test",
        [input_tensor],
        [output_tensor],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=8
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


def test_softplus_properties():
    """Test mathematical properties of Softplus."""
    # Softplus(x) = log(1 + exp(x))
    X = torch.randn(10, 20)

    softplus_output = torch.nn.functional.softplus(X)
    manual_output = torch.log(1 + torch.exp(X))

    # Note: For very large X, exp(X) overflows, so softplus uses approximation
    # Compare only for reasonable values
    mask = X < 10
    torch.testing.assert_close(
        softplus_output[mask], manual_output[mask], rtol=1e-5, atol=1e-5
    )

    # Softplus should always be positive
    assert (softplus_output > 0).all()

    # For large positive x, softplus(x) ≈ x
    large_x = torch.tensor([10.0, 20.0, 50.0])
    softplus_large = torch.nn.functional.softplus(large_x)
    torch.testing.assert_close(softplus_large, large_x, rtol=1e-2, atol=1e-2)

    # For large negative x, softplus(x) ≈ 0
    small_x = torch.tensor([-10.0, -20.0, -50.0])
    softplus_small = torch.nn.functional.softplus(small_x)
    assert (softplus_small < 0.01).all()


def test_softplus_vs_relu():
    """Test that Softplus is a smooth approximation of ReLU."""
    X = torch.linspace(-5, 5, 100)

    softplus_output = torch.nn.functional.softplus(X)
    relu_output = torch.nn.functional.relu(X)

    # Softplus should be close to ReLU for large positive values
    mask = X > 3
    torch.testing.assert_close(
        softplus_output[mask], relu_output[mask], rtol=0.1, atol=0.1
    )

    # Softplus should be smooth (no sharp corner at 0 like ReLU)
    # At x=0: softplus(0) = log(2) ≈ 0.693, relu(0) = 0
    softplus_at_zero = torch.nn.functional.softplus(torch.tensor([0.0]))
    assert abs(softplus_at_zero.item() - 0.693) < 0.01


def test_softplus_gradient():
    """Test that Softplus gradient is sigmoid."""
    # d/dx softplus(x) = sigmoid(x) = 1/(1 + exp(-x))
    X = torch.randn(5, 5, requires_grad=True)

    output = torch.nn.functional.softplus(X)
    output.sum().backward()

    # Gradient should be sigmoid(X)
    expected_grad = torch.sigmoid(X)

    torch.testing.assert_close(X.grad, expected_grad, rtol=1e-5, atol=1e-5)
