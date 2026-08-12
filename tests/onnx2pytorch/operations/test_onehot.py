import torch
import pytest
import numpy as np
from onnx.backend.test.case.node.onehot import one_hot

from onnx2pytorch.operations import OneHot
from tests.onnx2pytorch.differential import (
    assert_outputs_match,
    make_single_node_model,
    run_converted,
    run_onnxruntime,
)


@pytest.mark.parametrize("axis", [1, -2])
@pytest.mark.parametrize(
    "indices",
    [
        torch.tensor([[1, 9], [2, 4]], dtype=torch.float32),
        torch.tensor([0, 7, 8], dtype=torch.int64),
    ],
)
def test_onehot(indices, axis):
    on_value = 3
    off_value = 1
    output_type = torch.float32
    depth = torch.tensor([10], dtype=torch.float32)
    values = torch.tensor([off_value, on_value], dtype=output_type)
    # onnx's reference helper calls np.arange(depth); numpy>=2.5 rejects a
    # 1-d array here, so pass a python scalar.
    y = one_hot(indices.numpy(), int(depth), axis=axis, dtype=np.float32)
    y = y * (on_value - off_value) + off_value
    y = torch.tensor(y)

    op = OneHot(axis)
    out = op(indices, depth, values)
    assert torch.equal(y, out)


@pytest.mark.parametrize(
    "indices",
    [[1, 0], [-1, 0, -3], [5, 0], [-4, 3], [0, 1, 2], [-1, -2, -3]],
)
@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_onehot_index_range(indices, axis):
    """one_hot rejects a negative index, which onnx counts from the end, and an
    out of range one, which onnx leaves entirely off.

    onnxruntime is the oracle here: the onnx reference evaluator indexes with
    numpy and so wraps an out of range index around instead.
    """
    indices_array = np.array(indices, dtype=np.int64)
    depth = np.array(3, dtype=np.int64)
    values = np.array([0.0, 1.0], dtype=np.float32)
    inputs = {"indices": indices_array, "depth": depth, "values": values}
    model = make_single_node_model("OneHot", inputs, 11, axis=axis)
    assert_outputs_match(run_onnxruntime(model, inputs), run_converted(model, inputs))


@pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float64, np.int8])
def test_onehot_output_takes_the_type_of_values(dtype):
    indices = np.array([1, 0, 2], dtype=np.int64)
    depth = np.array(3, dtype=np.int64)
    values = np.array([0, 1], dtype=dtype)
    out = OneHot()(torch.tensor(indices), torch.tensor(depth), torch.tensor(values))
    assert out.numpy().dtype == dtype
