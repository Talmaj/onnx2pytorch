import numpy as np
import onnx
import pytest
import torch

from onnx2pytorch.convert.operations import convert_operations
from onnx2pytorch.operations import ReduceL2


@pytest.fixture
def tensor():
    return torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])


def test_reduce_l2_older_opset_version(tensor):
    shape = [3, 2, 2]
    axes = np.array([2], dtype=np.int64)
    keepdims = 0

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    op = ReduceL2(opset_version=10, keepdim=keepdims, dim=axes)

    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)


def test_do_not_keepdims_older_opset_version() -> None:
    opset_version = 10
    shape = [3, 2, 2]
    axes = np.array([2], dtype=np.int64)
    keepdims = 0

    node = onnx.helper.make_node(
        "ReduceL2",
        inputs=["data"],
        outputs=["reduced"],
        keepdims=keepdims,
        axes=axes,
    )
    graph = onnx.helper.make_graph([node], "test_reduce_l2_do_not_keepdims", [], [])

    ops = list(convert_operations(graph, opset_version))
    op = ops[0][2]

    assert isinstance(op, ReduceL2)

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )
    # print(reduced)
    # [[2.23606798, 5.],
    # [7.81024968, 10.63014581],
    # [13.45362405, 16.2788206]]

    out = op(torch.from_numpy(data))
    np.testing.assert_array_equal(out, reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    out = op(torch.from_numpy(data))
    np.testing.assert_array_equal(out, reduced)


def test_do_not_keepdims() -> None:
    shape = [3, 2, 2]
    axes = np.array([2], dtype=np.int64)
    keepdims = 0

    node = onnx.helper.make_node(
        "ReduceL2",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )
    graph = onnx.helper.make_graph([node], "test_reduce_l2_do_not_keepdims", [], [])
    ops = list(convert_operations(graph, 18))
    op = ops[0][2]

    assert isinstance(op, ReduceL2)

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )
    # print(reduced)
    # [[2.23606798, 5.],
    # [7.81024968, 10.63014581],
    # [13.45362405, 16.2788206]]

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)


def test_export_keepdims() -> None:
    shape = [3, 2, 2]
    axes = np.array([2], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceL2",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )
    graph = onnx.helper.make_graph([node], "test_reduce_l2_do_not_keepdims", [], [])
    ops = list(convert_operations(graph, 18))
    op = ops[0][2]

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )
    # print(reduced)
    # [[[2.23606798], [5.]]
    # [[7.81024968], [10.63014581]]
    # [[13.45362405], [16.2788206 ]]]

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)


def test_export_default_axes_keepdims() -> None:
    shape = [3, 2, 2]
    axes = np.array([], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceL2", inputs=["data", "axes"], outputs=["reduced"], keepdims=keepdims
    )
    graph = onnx.helper.make_graph([node], "test_reduce_l2_do_not_keepdims", [], [])
    ops = list(convert_operations(graph, 18))
    op = ops[0][2]

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

    reduced = np.sqrt(np.sum(a=np.square(data), axis=None, keepdims=keepdims == 1))
    # print(reduced)
    # [[[25.49509757]]]

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(np.sum(a=np.square(data), axis=None, keepdims=keepdims == 1))

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)


def test_export_negative_axes_keepdims() -> None:
    shape = [3, 2, 2]
    axes = np.array([-1], dtype=np.int64)
    keepdims = 1

    node = onnx.helper.make_node(
        "ReduceL2",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )
    graph = onnx.helper.make_graph([node], "test_reduce_l2_do_not_keepdims", [], [])
    ops = list(convert_operations(graph, 18))
    op = ops[0][2]

    data = np.reshape(np.arange(1, np.prod(shape) + 1, dtype=np.float32), shape)
    # print(data)
    # [[[1., 2.], [3., 4.]], [[5., 6.], [7., 8.]], [[9., 10.], [11., 12.]]]

    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )
    # print(reduced)
    # [[[2.23606798], [5.]]
    # [[7.81024968], [10.63014581]]
    # [[13.45362405], [16.2788206 ]]]

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)

    np.random.seed(0)
    data = np.random.uniform(-10, 10, shape).astype(np.float32)
    reduced = np.sqrt(
        np.sum(a=np.square(data), axis=tuple(axes), keepdims=keepdims == 1)
    )

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)


def test_export_empty_set() -> None:
    shape = [2, 0, 4]
    keepdims = 1
    reduced_shape = [2, 1, 4]

    node = onnx.helper.make_node(
        "ReduceL2",
        inputs=["data", "axes"],
        outputs=["reduced"],
        keepdims=keepdims,
    )
    graph = onnx.helper.make_graph([node], "test_reduce_l2_do_not_keepdims", [], [])
    ops = list(convert_operations(graph, 18))
    op = ops[0][2]

    data = np.array([], dtype=np.float32).reshape(shape)
    axes = np.array([1], dtype=np.int64)
    reduced = np.array(np.zeros(reduced_shape, dtype=np.float32))

    out = op(torch.from_numpy(data), axes=axes)
    np.testing.assert_array_equal(out, reduced)
