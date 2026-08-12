import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.bitshift import BitShift
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


def check_bitshift(x, y, direction):
    node = helper.make_node(
        "BitShift", inputs=["x", "y"], outputs=["z"], direction=direction
    )
    graph = helper.make_graph(
        [node],
        "bitshift_test",
        [
            helper.make_tensor_value_info("x", TensorProto.UINT8, list(x.shape)),
            helper.make_tensor_value_info("y", TensorProto.UINT8, list(y.shape)),
        ],
        [helper.make_tensor_value_info("z", TensorProto.UINT8, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_z = ort_session.run(None, {"x": x, "y": y})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        z = o2p_model(torch.from_numpy(x), torch.from_numpy(y))

    np.testing.assert_array_equal(z.numpy(), exp_z)


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
def test_convert_bitshift(direction):
    x = np.array([16, 4, 1], dtype=np.uint8)
    y = np.array([1, 2, 3], dtype=np.uint8)
    check_bitshift(x, y, direction)


@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
def test_convert_bitshift_broadcast(direction):
    np.random.seed(0)
    x = np.random.randint(0, 255, (3, 4)).astype(np.uint8)
    y = np.array([2], dtype=np.uint8)
    check_bitshift(x, y, direction)


def test_bitshift_left_uint8():
    op = BitShift(direction="LEFT")

    x = torch.tensor([16, 4, 1], dtype=torch.uint8)
    y = torch.tensor([1, 2, 3], dtype=torch.uint8)
    exp_z = torch.tensor([32, 16, 8], dtype=torch.uint8)
    assert torch.equal(op(x, y), exp_z)


def test_bitshift_left_int64():
    op = BitShift(direction="LEFT")

    x = torch.tensor([16, 4, 1], dtype=torch.int64)
    y = torch.tensor([1, 2, 3], dtype=torch.int64)
    exp_z = torch.tensor([32, 16, 8], dtype=torch.int64)
    assert torch.equal(op(x, y), exp_z)


def test_bitshift_right_uint8():
    op = BitShift(direction="RIGHT")

    x = torch.tensor([16, 4, 1], dtype=torch.uint8)
    y = torch.tensor([1, 2, 3], dtype=torch.uint8)
    exp_z = torch.tensor([8, 1, 0], dtype=torch.uint8)
    assert torch.equal(op(x, y), exp_z)


def test_bitshift_right_int64():
    op = BitShift(direction="RIGHT")

    x = torch.tensor([16, 4, 1], dtype=torch.int64)
    y = torch.tensor([1, 2, 3], dtype=torch.int64)
    exp_z = torch.tensor([8, 1, 0], dtype=torch.int64)
    assert torch.equal(op(x, y), exp_z)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("direction", ["LEFT", "RIGHT"])
def test_bitshift_unsigned_widths(dtype, direction):
    """Only uint8 shifts natively in torch, the wider types raised."""
    x = np.array([1, 2, 255, 7], dtype=dtype)
    y = np.array([1, 2, 1, 3], dtype=dtype)
    model = make_single_node_model(
        "BitShift", {"x": x, "y": y}, 11, direction=direction
    )
    assert_matches_oracle(model, {"x": x, "y": y})
