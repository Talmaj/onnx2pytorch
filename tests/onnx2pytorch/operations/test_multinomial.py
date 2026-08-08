import numpy as np
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def run(x, out_type=TensorProto.INT32, **attrs):
    node = helper.make_node("Multinomial", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "multinomial_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 7)])
    with torch.no_grad():
        return ConvertModel(model)(torch.from_numpy(x))


def test_multinomial_shape_and_dtype():
    x = np.log(np.array([[0.2, 0.3, 0.5], [0.1, 0.1, 0.8]], dtype=np.float32))
    y = run(x, sample_size=10)

    assert y.shape == (2, 10)
    assert y.dtype == torch.int32
    assert torch.all((y >= 0) & (y < 3))


def test_multinomial_default_sample_size():
    x = np.log(np.array([[0.5, 0.5]], dtype=np.float32))
    assert run(x).shape == (1, 1)


def test_multinomial_int64_dtype():
    x = np.log(np.array([[0.5, 0.5]], dtype=np.float32))
    y = run(x, out_type=TensorProto.INT64, dtype=TensorProto.INT64, sample_size=4)
    assert y.dtype == torch.int64


def test_multinomial_picks_only_possible_outcome():
    x = np.array([[-1000.0, 0.0, -1000.0]], dtype=np.float32)
    y = run(x, sample_size=50)
    np.testing.assert_array_equal(y.numpy(), np.ones((1, 50), dtype=np.int32))


def test_multinomial_distribution():
    probabilities = np.array([0.1, 0.6, 0.3], dtype=np.float32)
    x = np.log(probabilities)[None, :]
    y = run(x, sample_size=20000, seed=3.0)

    counts = np.bincount(y.numpy().ravel(), minlength=3) / y.numel()
    np.testing.assert_allclose(counts, probabilities, atol=0.02)


def test_multinomial_seed_is_reproducible():
    x = np.log(np.array([[0.25, 0.25, 0.25, 0.25]], dtype=np.float32))
    np.testing.assert_array_equal(
        run(x, sample_size=50, seed=11.0).numpy(),
        run(x, sample_size=50, seed=11.0).numpy(),
    )


def test_multinomial_unnormalized_log_probabilities():
    # An overall shift of the log probabilities must not change the distribution
    probabilities = np.array([0.2, 0.8], dtype=np.float32)
    x = (np.log(probabilities) + 7.0)[None, :]
    y = run(x, sample_size=20000, seed=5.0)

    counts = np.bincount(y.numpy().ravel(), minlength=2) / y.numel()
    np.testing.assert_allclose(counts, probabilities, atol=0.02)
