import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)

# ReduceSum moved axes from attribute to input at opset 13, all the others at 18
ATTRIBUTE_AXES = [
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
]


@pytest.fixture
def x():
    np.random.seed(0)
    return (np.abs(np.random.randn(2, 3, 4)) + 0.5).astype(np.float32)


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("opset_version", [11, 13, 17])
@pytest.mark.parametrize("op_type", ATTRIBUTE_AXES)
def test_reduction_axes_attribute(x, op_type, opset_version, keepdims):
    """axes stays an attribute up to opset 17 for everything but ReduceSum."""
    model = make_single_node_model(
        op_type, {"x": x}, opset_version, axes=[0, 2], keepdims=keepdims
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("opset_version", [18, 20])
@pytest.mark.parametrize("op_type", ATTRIBUTE_AXES)
def test_reduction_axes_input(x, op_type, opset_version, keepdims):
    axes = np.array([0, 2], dtype=np.int64)
    model = make_single_node_model(
        op_type,
        {"x": x},
        opset_version,
        initializers={"axes": axes},
        keepdims=keepdims,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("op_type", ATTRIBUTE_AXES + ["ReduceSum"])
def test_reduction_over_all_axes(x, op_type):
    model = make_single_node_model(op_type, {"x": x}, 18, keepdims=0)
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("op_type", ATTRIBUTE_AXES + ["ReduceSum"])
def test_reduction_noop_with_empty_axes(op_type):
    """A skipped reduction still applies the operator's element-wise part."""
    data = np.array([[-1.5, 2.0], [0.5, -3.0]], dtype=np.float32)
    model = make_single_node_model(
        op_type, {"x": data}, 18, keepdims=1, noop_with_empty_axes=1
    )
    assert_matches_oracle(model, {"x": data})


@pytest.mark.parametrize("op_type", ["ReduceMin", "ReduceMean", "ReduceProd"])
def test_reduction_single_output(x, op_type):
    """ReduceMin used to leak torch.min's indices as a second output."""
    model = make_single_node_model(op_type, {"x": x}, 13, axes=[1], keepdims=1)
    assert len(assert_matches_oracle(model, {"x": x})) == 1
