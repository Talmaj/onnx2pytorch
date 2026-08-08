import numpy as np
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def run(x, in_type=TensorProto.FLOAT, out_type=TensorProto.FLOAT, **attrs):
    node = helper.make_node("RandomNormalLike", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "randomnormallike_test",
        [helper.make_tensor_value_info("x", in_type, list(x.shape))],
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    with torch.no_grad():
        return ConvertModel(model)(torch.from_numpy(x))


def test_random_normal_like_shape_and_dtype():
    x = np.zeros((3, 4, 5), dtype=np.float32)
    y = run(x)

    assert y.shape == x.shape
    assert y.dtype == torch.float32


def test_random_normal_like_distribution():
    x = np.zeros((50000,), dtype=np.float32)
    y = run(x, seed=1.0)

    assert abs(y.mean().item()) < 0.05
    assert abs(y.std().item() - 1.0) < 0.05


def test_random_normal_like_mean_and_scale():
    x = np.zeros((50000,), dtype=np.float32)
    y = run(x, mean=-3.0, scale=0.5, seed=2.0)

    assert abs(y.mean().item() + 3.0) < 0.05
    assert abs(y.std().item() - 0.5) < 0.05


def test_random_normal_like_seed_is_reproducible():
    x = np.zeros((20,), dtype=np.float32)
    np.testing.assert_array_equal(run(x, seed=42.0).numpy(), run(x, seed=42.0).numpy())


def test_random_normal_like_different_seeds():
    x = np.zeros((20,), dtype=np.float32)
    assert not torch.equal(run(x, seed=1.0), run(x, seed=2.0))


def test_random_normal_like_keeps_input_dtype():
    x = np.zeros((2, 3), dtype=np.float64)
    y = run(x, in_type=TensorProto.DOUBLE, out_type=TensorProto.DOUBLE, seed=3.0)
    assert y.dtype == torch.float64


def test_random_normal_like_explicit_dtype():
    x = np.zeros((2, 3), dtype=np.float64)
    y = run(
        x,
        in_type=TensorProto.DOUBLE,
        out_type=TensorProto.FLOAT,
        dtype=TensorProto.FLOAT,
        seed=4.0,
    )
    assert y.dtype == torch.float32
