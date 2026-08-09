import io
import numpy as np
import onnx
import pytest
import torch

from onnx2pytorch.convert import ConvertModel
from torch import nn

from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
)


class UsedIndices(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp = nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=[2, 2],
            ceil_mode=True,
            return_indices=True,
        )

    def forward(self, x):
        y, indices = self.mp(x)
        return y - 42, indices + 42


class UnusedIndices(nn.Module):
    def __init__(self):
        super().__init__()
        self.mp = nn.MaxPool2d(
            kernel_size=[3, 3],
            stride=[2, 2],
            ceil_mode=True,
        )

    def forward(self, x):
        return self.mp(x) - 42


def test_maxpool_2d_ceil():
    x = torch.tensor(
        [
            [
                [
                    [1, 2, 3, 4],
                    [5, 6, 7, 8],
                    [9, 10, 11, 12],
                    [13, 14, 15, 16],
                ]
            ]
        ],
        dtype=torch.float32,
    )
    exp_y = (
        torch.tensor(
            [
                [
                    [
                        [11, 12],
                        [15, 16],
                    ]
                ]
            ],
            dtype=torch.float32,
        )
        - 42
    )
    exp_indices = (
        torch.tensor(
            [
                [
                    [
                        [10, 11],
                        [14, 15],
                    ]
                ]
            ]
        )
        + 42
    )

    model = UsedIndices()
    bitstream = io.BytesIO()
    torch.onnx.export(
        model=model,
        args=(x,),
        f=bitstream,
        input_names=["x"],
        opset_version=11,
        dynamo=False,
    )
    onnx_model = onnx.ModelProto.FromString(bitstream.getvalue())
    o2p_model = ConvertModel(onnx_model)
    y, indices = o2p_model(x)
    assert torch.equal(exp_y, y)
    assert torch.equal(exp_indices, indices)

    model = UnusedIndices()
    bitstream = io.BytesIO()
    torch.onnx.export(
        model=model,
        args=(x,),
        f=bitstream,
        input_names=["x"],
        opset_version=11,
        dynamo=False,
    )
    onnx_model = onnx.ModelProto.FromString(bitstream.getvalue())
    o2p_model = ConvertModel(onnx_model)
    y = o2p_model(x)
    assert torch.equal(exp_y, y)


@pytest.mark.parametrize("opset_version", [8, 10, 11, 12, 22])
def test_maxpool_indices_span_channels(opset_version):
    """ONNX indices are flattened over the whole tensor, torch's are per plane."""
    np.random.seed(0)
    x = np.random.randn(2, 3, 5, 5).astype(np.float32)
    model = make_single_node_model(
        "MaxPool",
        {"x": x},
        opset_version,
        outputs=("y", "i"),
        kernel_shape=[2, 2],
        strides=[2, 2],
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("storage_order", [0, 1])
def test_maxpool_storage_order(storage_order):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 6).astype(np.float32)
    model = make_single_node_model(
        "MaxPool",
        {"x": x},
        12,
        outputs=("y", "i"),
        kernel_shape=[2, 2],
        strides=[2, 2],
        storage_order=storage_order,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("kernel_shape", [[2], [2, 2], [2, 2, 2]])
def test_maxpool_indices_by_rank(kernel_shape):
    np.random.seed(0)
    x = np.random.randn(*([2, 3] + [4] * len(kernel_shape))).astype(np.float32)
    model = make_single_node_model(
        "MaxPool",
        {"x": x},
        12,
        outputs=("y", "i"),
        kernel_shape=kernel_shape,
        strides=[2] * len(kernel_shape),
    )
    assert_matches_oracle(model, {"x": x})
