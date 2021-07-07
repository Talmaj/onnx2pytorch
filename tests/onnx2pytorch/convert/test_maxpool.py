import io
import onnx
import pytest
import torch

from onnx2pytorch.convert import ConvertModel
from torch import nn


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
    )
    onnx_model = onnx.ModelProto.FromString(bitstream.getvalue())
    o2p_model = ConvertModel(onnx_model)
    y = o2p_model(x)
    assert torch.equal(exp_y, y)
