"""Cross cutting reduction behaviour: the output type and the empty axes list."""

import numpy as np
import pytest

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)

# Every reduction that takes axes as an input, with the opset that introduced it
AXES_AS_INPUT = {
    "ReduceSum": 13,
    "ReduceL1": 18,
    "ReduceL2": 18,
    "ReduceLogSum": 18,
    "ReduceLogSumExp": 18,
    "ReduceMax": 18,
    "ReduceMean": 18,
    "ReduceMin": 18,
    "ReduceProd": 18,
    "ReduceSumSquare": 18,
}

# The reductions that stay exact on integers, so onnx keeps the input type
INTEGER_REDUCTIONS = [
    "ReduceSum",
    "ReduceL1",
    "ReduceL2",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
]


@pytest.mark.parametrize("op_type,opset_version", sorted(AXES_AS_INPUT.items()))
@pytest.mark.parametrize("noop_with_empty_axes", [0, 1])
def test_empty_axes_tensor(op_type, opset_version, noop_with_empty_axes):
    """An explicitly empty axes tensor is not the same as an absent one, and the
    reductions that resolved axes by hand used to reduce over everything."""
    np.random.seed(0)
    data = np.random.randn(2, 3).astype(np.float32)
    model = make_single_node_model(
        op_type,
        {"data": data},
        opset_version,
        initializers={"axes": np.array([], dtype=np.int64)},
        noop_with_empty_axes=noop_with_empty_axes,
    )
    assert_matches_oracle(model, {"data": data})


@pytest.mark.parametrize("op_type", INTEGER_REDUCTIONS)
@pytest.mark.parametrize("keepdims", [0, 1])
def test_integer_output_keeps_the_input_type(op_type, keepdims):
    """torch accumulates integers into int64 and takes roots and means in
    floating point, while onnx returns the input type."""
    data = np.array([[1, 2, 3], [-4, 5, -7]], dtype=np.int32)
    model = make_single_node_model(
        op_type, {"data": data}, AXES_AS_INPUT[op_type], keepdims=keepdims
    )
    assert_matches_oracle(model, {"data": data})


@pytest.mark.parametrize("op_type", INTEGER_REDUCTIONS)
def test_integer_output_along_one_axis(op_type):
    data = np.array([[1, 2, 3], [-4, 5, -7]], dtype=np.int32)
    model = make_single_node_model(
        op_type,
        {"data": data},
        AXES_AS_INPUT[op_type],
        initializers={"axes": np.array([0], dtype=np.int64)},
    )
    assert_matches_oracle(model, {"data": data})


@pytest.mark.parametrize("op_type", ["CumSum", "CumProd"])
def test_cumulative_integer_output_keeps_the_input_type(op_type):
    data = np.array([1, 2, 3, -4], dtype=np.int32)
    axis = np.array(0, dtype=np.int64)
    opset_version = 14 if op_type == "CumSum" else 26
    model = make_single_node_model(op_type, {"data": data, "axis": axis}, opset_version)
    assert_matches_oracle(model, {"data": data, "axis": axis})
