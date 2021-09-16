import pytest
import torch

from onnx2pytorch.operations.bitshift import BitShift


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
