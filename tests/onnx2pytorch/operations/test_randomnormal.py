import numpy as np
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def run(out_type=TensorProto.FLOAT, **attrs):
    node = helper.make_node("RandomNormal", inputs=[], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "randomnormal_test",
        [],
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    with torch.no_grad():
        return ConvertModel(model)()


def test_random_normal_shape_and_dtype():
    y = run(shape=[3, 4, 5])

    assert y.shape == (3, 4, 5)
    assert y.dtype == torch.float32


def test_random_normal_distribution():
    y = run(shape=[50000], seed=1.0)

    assert abs(y.mean().item()) < 0.05
    assert abs(y.std().item() - 1.0) < 0.05


def test_random_normal_mean_and_scale():
    y = run(shape=[50000], mean=5.0, scale=2.0, seed=2.0)

    assert abs(y.mean().item() - 5.0) < 0.1
    assert abs(y.std().item() - 2.0) < 0.1


def test_random_normal_seed_is_reproducible():
    np.testing.assert_array_equal(
        run(shape=[20], seed=42.0).numpy(), run(shape=[20], seed=42.0).numpy()
    )


def test_random_normal_different_seeds():
    assert not torch.equal(run(shape=[20], seed=1.0), run(shape=[20], seed=2.0))


def test_random_normal_double_dtype():
    y = run(
        out_type=TensorProto.DOUBLE, shape=[2, 3], dtype=TensorProto.DOUBLE, seed=3.0
    )

    assert y.shape == (2, 3)
    assert y.dtype == torch.float64
