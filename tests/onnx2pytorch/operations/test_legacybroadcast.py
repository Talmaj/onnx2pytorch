import numpy as np
import pytest

from tests.onnx2pytorch.differential import make_single_node_model, run_converted

# Neither onnxruntime nor onnx's reference evaluator implements the pre-7
# broadcast/axis attributes, so the expectation is spelled out here.
NUMPY_OPS = {
    "Add": np.add,
    "Sub": np.subtract,
    "Mul": np.multiply,
    "Div": np.divide,
}


@pytest.mark.parametrize("op_type", sorted(NUMPY_OPS))
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_legacy_broadcast_axis(op_type, axis):
    """B aligns with A at axis, not at the last dimension."""
    np.random.seed(0)
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(*a.shape[axis : axis + 2]).astype(np.float32) + 2.0
    inputs = {"a": a, "b": b}
    model = make_single_node_model("Add", inputs, 6, broadcast=1, axis=axis)
    model.graph.node[0].op_type = op_type

    expected = NUMPY_OPS[op_type](
        a, b.reshape(b.shape + (1,) * (a.ndim - axis - b.ndim))
    )
    np.testing.assert_allclose(
        run_converted(model, inputs)[0], expected, rtol=1e-5, atol=1e-6
    )


def test_legacy_broadcast_is_wrong_without_axis_handling():
    """A square trailing shape hides the misalignment unless axis is honoured."""
    np.random.seed(0)
    a = np.random.randn(3, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    inputs = {"a": a, "b": b}
    model = make_single_node_model("Add", inputs, 6, broadcast=1, axis=0)

    np.testing.assert_allclose(
        run_converted(model, inputs)[0], a + b.reshape(3, 1), rtol=1e-5, atol=1e-6
    )


@pytest.mark.parametrize("op_type", sorted(NUMPY_OPS))
def test_legacy_broadcast_without_axis_matches_numpy(op_type):
    """Without axis the legacy rules coincide with numpy's suffix matching."""
    np.random.seed(0)
    a = np.random.randn(2, 3, 4, 5).astype(np.float32)
    b = np.random.randn(4, 5).astype(np.float32) + 2.0
    inputs = {"a": a, "b": b}
    model = make_single_node_model("Add", inputs, 6, broadcast=1)
    model.graph.node[0].op_type = op_type

    np.testing.assert_allclose(
        run_converted(model, inputs)[0],
        NUMPY_OPS[op_type](a, b),
        rtol=1e-5,
        atol=1e-6,
    )
