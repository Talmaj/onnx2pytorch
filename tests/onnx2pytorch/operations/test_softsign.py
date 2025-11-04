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
        [3, 3, 3, 3],
    ],
)
def test_softsign_onnxruntime(input_shape):
    """Test Softsign against onnxruntime."""
    np.random.seed(42)

    # Create input with varied values (positive, negative, zero)
    X = (
        np.random.randn(*input_shape).astype(np.float32) * 5
    )  # Scale to get larger values

    # Create ONNX graph with Softsign node
    input_tensor = helper.make_tensor_value_info("X", TensorProto.FLOAT, input_shape)
    output_tensor = helper.make_tensor_value_info("Y", TensorProto.FLOAT, input_shape)

    softsign_node = helper.make_node(
        "Softsign",
        inputs=["X"],
        outputs=["Y"],
    )

    graph = helper.make_graph(
        [softsign_node],
        "softsign_test",
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


def test_softsign_formula():
    """Test that Softsign implements x / (1 + |x|)."""
    X = torch.tensor([-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])

    softsign_output = torch.nn.functional.softsign(X)
    manual_output = X / (1 + torch.abs(X))

    torch.testing.assert_close(softsign_output, manual_output, rtol=1e-6, atol=1e-6)


def test_softsign_properties():
    """Test mathematical properties of Softsign."""
    X = torch.linspace(-10, 10, 100)

    softsign_output = torch.nn.functional.softsign(X)

    # Softsign output should be in range (-1, 1)
    assert (softsign_output > -1).all()
    assert (softsign_output < 1).all()

    # Softsign(0) = 0
    zero_output = torch.nn.functional.softsign(torch.tensor([0.0]))
    assert abs(zero_output.item()) < 1e-6

    # Softsign is odd function: softsign(-x) = -softsign(x)
    X_test = torch.tensor([1.0, 2.0, 3.0, 5.0])
    softsign_pos = torch.nn.functional.softsign(X_test)
    softsign_neg = torch.nn.functional.softsign(-X_test)
    torch.testing.assert_close(softsign_neg, -softsign_pos, rtol=1e-6, atol=1e-6)

    # For large |x|, softsign(x) approaches ±1
    large_pos = torch.nn.functional.softsign(torch.tensor([100.0]))
    large_neg = torch.nn.functional.softsign(torch.tensor([-100.0]))
    assert abs(large_pos.item() - 1.0) < 0.01
    assert abs(large_neg.item() + 1.0) < 0.01


def test_softsign_vs_tanh():
    """Test that Softsign is similar to tanh but with different shape."""
    X = torch.linspace(-5, 5, 100)

    softsign_output = torch.nn.functional.softsign(X)
    tanh_output = torch.tanh(X)

    # Both should be in (-1, 1)
    assert (softsign_output > -1).all() and (softsign_output < 1).all()
    assert (tanh_output > -1).all() and (tanh_output < 1).all()

    # Both should be odd functions passing through origin
    # Find the index closest to x=0
    zero_idx = X.abs().argmin()
    # At x≈0, both should be close to 0
    assert abs(softsign_output[zero_idx].item()) < 0.1
    assert abs(tanh_output[zero_idx].item()) < 0.1

    # Softsign approaches asymptotes more slowly than tanh
    # At x=5: tanh(5) ≈ 0.9999, softsign(5) = 5/6 ≈ 0.833
    assert abs(tanh_output[-1].item() - 1.0) < abs(softsign_output[-1].item() - 1.0)


def test_softsign_gradient():
    """Test Softsign gradient: d/dx softsign(x) = 1 / (1 + |x|)^2."""
    X = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], requires_grad=True)

    output = torch.nn.functional.softsign(X)
    output.sum().backward()

    # Expected gradient
    expected_grad = 1.0 / (1.0 + torch.abs(X)) ** 2

    torch.testing.assert_close(X.grad, expected_grad, rtol=1e-5, atol=1e-5)


def test_softsign_extreme_values():
    """Test Softsign with extreme input values."""
    # Very large positive (but not so large that floating point precision makes it exactly 1.0)
    large_pos = torch.tensor([100.0, 1000.0, 10000.0])
    output_pos = torch.nn.functional.softsign(large_pos)
    assert (
        output_pos <= 1.0
    ).all()  # Use <= since at extreme values it can be exactly 1.0
    assert (output_pos > 0.99).all()  # Should be very close to 1

    # Very large negative
    large_neg = torch.tensor([-100.0, -1000.0, -10000.0])
    output_neg = torch.nn.functional.softsign(large_neg)
    assert (
        output_neg >= -1.0
    ).all()  # Use >= since at extreme values it can be exactly -1.0
    assert (output_neg < -0.99).all()  # Should be very close to -1

    # Very small (near zero)
    small = torch.tensor([1e-6, -1e-6, 1e-10])
    output_small = torch.nn.functional.softsign(small)
    # For small x, softsign(x) ≈ x
    torch.testing.assert_close(output_small, small, rtol=1e-3, atol=1e-3)
