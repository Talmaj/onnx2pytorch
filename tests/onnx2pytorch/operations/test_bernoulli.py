import numpy as np
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_model(x, out_type=TensorProto.FLOAT, **attrs):
    node = helper.make_node("Bernoulli", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "bernoulli_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 15)])


def run(x, out_type=TensorProto.FLOAT, **attrs):
    model = build_model(x, out_type, **attrs)
    with torch.no_grad():
        return ConvertModel(model)(torch.from_numpy(x))


def test_bernoulli_shape_and_values():
    np.random.seed(0)
    x = np.random.rand(4, 5).astype(np.float32)
    y = run(x)

    assert y.shape == x.shape
    assert y.dtype == torch.float32
    assert torch.all((y == 0) | (y == 1))


def test_bernoulli_deterministic_probabilities():
    x = np.array([[0.0, 1.0, 0.0, 1.0]], dtype=np.float32)
    y = run(x)
    np.testing.assert_array_equal(y.numpy(), x)


def test_bernoulli_distribution():
    x = np.full((20000,), 0.3, dtype=np.float32)
    y = run(x, seed=7.0)
    assert abs(y.mean().item() - 0.3) < 0.02


def test_bernoulli_seed_is_reproducible():
    np.random.seed(1)
    x = np.random.rand(100).astype(np.float32)
    np.testing.assert_array_equal(run(x, seed=42.0).numpy(), run(x, seed=42.0).numpy())


def test_bernoulli_different_seeds():
    np.random.seed(2)
    x = np.full((200,), 0.5, dtype=np.float32)
    assert not torch.equal(run(x, seed=1.0), run(x, seed=2.0))


def test_bernoulli_dtype():
    np.random.seed(3)
    x = np.random.rand(3, 4).astype(np.float32)
    y = run(x, out_type=TensorProto.INT32, dtype=TensorProto.INT32)

    assert y.dtype == torch.int32
    assert torch.all((y == 0) | (y == 1))
