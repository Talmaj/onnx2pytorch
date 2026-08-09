import numpy as np
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def run(out_type=TensorProto.FLOAT, **attrs):
    node = helper.make_node("RandomUniform", inputs=[], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "randomuniform_test",
        [],
        [helper.make_tensor_value_info("y", out_type, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    with torch.no_grad():
        return ConvertModel(model)()


def test_random_uniform_shape_and_dtype():
    y = run(shape=[3, 4, 5])

    assert y.shape == (3, 4, 5)
    assert y.dtype == torch.float32
    assert torch.all((y >= 0.0) & (y < 1.0))


def test_random_uniform_range():
    y = run(shape=[10000], low=-5.0, high=10.0, seed=1.0)

    assert torch.all((y >= -5.0) & (y < 10.0))
    assert abs(y.mean().item() - 2.5) < 0.2


def test_random_uniform_distribution():
    y = run(shape=[20000], seed=2.0)

    counts = torch.histc(y, bins=10, min=0.0, max=1.0)
    assert torch.all(counts > 1500)


def test_random_uniform_seed_is_reproducible():
    np.testing.assert_array_equal(
        run(shape=[20], seed=42.0).numpy(), run(shape=[20], seed=42.0).numpy()
    )


def test_random_uniform_different_seeds():
    assert not torch.equal(run(shape=[20], seed=1.0), run(shape=[20], seed=2.0))


def test_random_uniform_double_dtype():
    y = run(
        out_type=TensorProto.DOUBLE, shape=[2, 3], dtype=TensorProto.DOUBLE, seed=3.0
    )

    assert y.shape == (2, 3)
    assert y.dtype == torch.float64
