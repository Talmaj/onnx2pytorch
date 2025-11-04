import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.batchnorm import BatchNormWrapper


@pytest.mark.parametrize(
    "batch_size,channels,height,width,epsilon,momentum",
    [
        # Test with batch_size=1
        (1, 3, 5, 5, 1e-5, 0.9),
        # Test with batch_size>1 (the critical case)
        (2, 3, 5, 5, 1e-5, 0.9),
        (4, 3, 5, 5, 1e-5, 0.9),
        (8, 16, 7, 7, 1e-5, 0.9),
        # Test with different epsilons
        (2, 3, 5, 5, 1e-3, 0.9),
        (2, 3, 5, 5, 1e-7, 0.9),
        # Test with different momentums
        (2, 3, 5, 5, 1e-5, 0.1),
        (2, 3, 5, 5, 1e-5, 0.99),
        # Test with different spatial dimensions
        (2, 8, 10, 10, 1e-5, 0.9),
        (2, 16, 3, 3, 1e-5, 0.9),
    ],
)
def test_batchnorm_onnxruntime(batch_size, channels, height, width, epsilon, momentum):
    """Test BatchNorm against onnxruntime with various batch sizes."""
    np.random.seed(42)
    torch.manual_seed(42)

    # Create input
    X = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # Create BatchNorm parameters
    scale = np.random.randn(channels).astype(np.float32)
    bias = np.random.randn(channels).astype(np.float32)
    mean = np.random.randn(channels).astype(np.float32)
    var = np.abs(np.random.randn(channels).astype(np.float32)) + 0.1  # Ensure positive

    # Create ONNX graph with BatchNormalization node
    input_tensor = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch_size, channels, height, width]
    )
    output_tensor = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch_size, channels, height, width]
    )

    scale_init = helper.make_tensor(
        "scale", TensorProto.FLOAT, [channels], scale.tolist()
    )
    bias_init = helper.make_tensor("B", TensorProto.FLOAT, [channels], bias.tolist())
    mean_init = helper.make_tensor("mean", TensorProto.FLOAT, [channels], mean.tolist())
    var_init = helper.make_tensor("var", TensorProto.FLOAT, [channels], var.tolist())

    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["X", "scale", "B", "mean", "var"],
        outputs=["Y"],
        epsilon=epsilon,
        momentum=momentum,
    )

    graph = helper.make_graph(
        [bn_node],
        "batchnorm_test",
        [input_tensor],
        [output_tensor],
        [scale_init, bias_init, mean_init, var_init],
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
        msg=f"BatchNorm mismatch for batch_size={batch_size}, channels={channels}",
    )


def test_batchnorm_bias_fix():
    """Test that the bias parameter is correctly applied (not overwritten by scale)."""
    np.random.seed(42)

    batch_size = 2
    channels = 4
    height, width = 5, 5

    X = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    # Create BatchNorm parameters with distinct scale and bias
    scale = np.ones(channels, dtype=np.float32) * 2.0  # Scale = 2
    bias = np.ones(channels, dtype=np.float32) * 5.0  # Bias = 5 (should NOT be 2!)
    mean = np.zeros(channels, dtype=np.float32)
    var = np.ones(channels, dtype=np.float32)

    # Create ONNX model
    input_tensor = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch_size, channels, height, width]
    )
    output_tensor = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch_size, channels, height, width]
    )

    scale_init = helper.make_tensor(
        "scale", TensorProto.FLOAT, [channels], scale.tolist()
    )
    bias_init = helper.make_tensor("B", TensorProto.FLOAT, [channels], bias.tolist())
    mean_init = helper.make_tensor("mean", TensorProto.FLOAT, [channels], mean.tolist())
    var_init = helper.make_tensor("var", TensorProto.FLOAT, [channels], var.tolist())

    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["X", "scale", "B", "mean", "var"],
        outputs=["Y"],
        epsilon=1e-5,
    )

    graph = helper.make_graph(
        [bn_node],
        "batchnorm_bias_test",
        [input_tensor],
        [output_tensor],
        [scale_init, bias_init, mean_init, var_init],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=8
    )

    # Run with onnxruntime (ground truth)
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_outputs = ort_session.run(None, {"X": X})
    expected_Y = ort_outputs[0]

    # Convert to PyTorch
    o2p_model = ConvertModel(model, experimental=True)
    X_torch = torch.from_numpy(X)

    with torch.no_grad():
        o2p_output = o2p_model(X_torch)

    # If bias was incorrectly set to scale (the bug), outputs would differ
    torch.testing.assert_close(
        o2p_output,
        torch.from_numpy(expected_Y),
        rtol=1e-5,
        atol=1e-5,
        msg="Bias parameter was not correctly applied",
    )

    # Verify that the output includes the bias (should be around 5, not 2)
    # After normalization: (X - 0) / sqrt(1 + eps) * 2 + 5 ≈ X * 2 + 5
    # The mean should be around 5 (from bias), not 2 (from scale)
    output_mean_per_channel = o2p_output.mean(dim=(0, 2, 3))
    # The mean should be close to bias (5), not scale (2)
    # Note: This is approximate since X is random
    assert torch.allclose(
        output_mean_per_channel, torch.tensor([5.0] * channels), rtol=1, atol=1
    )


def test_batchnorm_eval_mode():
    """Test that BatchNorm uses eval mode (running statistics)."""

    channels = 4
    scale = torch.ones(channels)
    bias = torch.zeros(channels)
    running_mean = torch.randn(channels)
    running_var = torch.abs(torch.randn(channels)) + 0.1

    # Create BatchNormWrapper
    bn_wrapper = BatchNormWrapper([scale, bias, running_mean, running_var])

    # Verify it's in eval mode
    assert not bn_wrapper.bnu.training, "BatchNorm should be in eval mode"

    # Test with batch_size > 1
    X = torch.randn(4, channels, 5, 5)

    output = bn_wrapper(X)

    # In eval mode, it should use running_mean and running_var,
    # not compute statistics from the current batch
    # Verify output shape
    assert output.shape == X.shape


def test_batchnorm_formula():
    """Test that BatchNorm implements the correct formula."""
    batch_size = 2
    channels = 3
    height, width = 4, 4

    X = torch.randn(batch_size, channels, height, width)

    scale = torch.ones(channels) * 2.0
    bias = torch.ones(channels) * 3.0
    mean = torch.zeros(channels)
    var = torch.ones(channels)
    epsilon = 1e-5

    # Manual computation: Y = scale * (X - mean) / sqrt(var + epsilon) + bias
    expected = scale.view(1, -1, 1, 1) * (X - mean.view(1, -1, 1, 1)) / torch.sqrt(
        var.view(1, -1, 1, 1) + epsilon
    ) + bias.view(1, -1, 1, 1)

    # Using BatchNormWrapper

    bn_wrapper = BatchNormWrapper([scale, bias, mean, var], eps=epsilon)
    output = bn_wrapper(X)

    torch.testing.assert_close(output, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
def test_batchnorm_consistency_across_batch_sizes(batch_size):
    """Test that BatchNorm produces consistent results across different batch sizes."""
    np.random.seed(42)
    torch.manual_seed(42)

    channels = 8
    height, width = 6, 6

    # Create a deterministic input pattern
    X = np.random.randn(batch_size, channels, height, width).astype(np.float32)

    scale = np.random.randn(channels).astype(np.float32)
    bias = np.random.randn(channels).astype(np.float32)
    mean = np.random.randn(channels).astype(np.float32)
    var = np.abs(np.random.randn(channels).astype(np.float32)) + 0.1

    # Create ONNX model
    input_tensor = helper.make_tensor_value_info(
        "X", TensorProto.FLOAT, [batch_size, channels, height, width]
    )
    output_tensor = helper.make_tensor_value_info(
        "Y", TensorProto.FLOAT, [batch_size, channels, height, width]
    )

    scale_init = helper.make_tensor(
        "scale", TensorProto.FLOAT, [channels], scale.tolist()
    )
    bias_init = helper.make_tensor("B", TensorProto.FLOAT, [channels], bias.tolist())
    mean_init = helper.make_tensor("mean", TensorProto.FLOAT, [channels], mean.tolist())
    var_init = helper.make_tensor("var", TensorProto.FLOAT, [channels], var.tolist())

    bn_node = helper.make_node(
        "BatchNormalization",
        inputs=["X", "scale", "B", "mean", "var"],
        outputs=["Y"],
        epsilon=1e-5,
    )

    graph = helper.make_graph(
        [bn_node],
        "batchnorm_consistency_test",
        [input_tensor],
        [output_tensor],
        [scale_init, bias_init, mean_init, var_init],
    )

    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 11)], ir_version=8
    )

    # Run with onnxruntime
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_outputs = ort_session.run(None, {"X": X})
    expected_Y = ort_outputs[0]

    # Convert to PyTorch
    o2p_model = ConvertModel(model, experimental=True)
    X_torch = torch.from_numpy(X)

    with torch.no_grad():
        o2p_output = o2p_model(X_torch)

    # Should match onnxruntime regardless of batch size
    torch.testing.assert_close(
        o2p_output,
        torch.from_numpy(expected_Y),
        rtol=1e-5,
        atol=1e-5,
        msg=f"BatchNorm failed for batch_size={batch_size}",
    )
