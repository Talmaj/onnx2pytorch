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


@pytest.mark.parametrize(
    "pads,storage_order",
    [([1, 1, 0, 0], 0), ([1, 1, 0, 0], 1), ([0, 1, 1, 0], 0), ([1, 1, 1, 1], 0)],
)
def test_maxpool_asymmetric_pads_are_not_zeros(pads, storage_order):
    """A materialised pad used to be filled with zeros, which then won every
    window of a negative input, and the indices counted from the padded plane."""
    x = -np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    model = make_single_node_model(
        "MaxPool",
        {"x": x},
        12,
        outputs=("y", "indices"),
        kernel_shape=[2, 2],
        pads=pads,
        storage_order=storage_order,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("auto_pad", ["SAME_UPPER", "SAME_LOWER"])
@pytest.mark.parametrize("kernel_shape,strides", [([2, 2], [1, 1]), ([3, 3], [2, 2])])
def test_maxpool_auto_pad_indices(auto_pad, kernel_shape, strides):
    x = -np.arange(1, 10, dtype=np.float32).reshape(1, 1, 3, 3)
    model = make_single_node_model(
        "MaxPool",
        {"x": x},
        12,
        outputs=("y", "indices"),
        kernel_shape=kernel_shape,
        strides=strides,
        auto_pad=auto_pad,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("spatial", [1, 3])
def test_maxpool_padded_indices_over_other_ranks(spatial):
    np.random.seed(0)
    shape = (2, 3) + (4,) * spatial
    x = np.random.randn(*shape).astype(np.float32)
    model = make_single_node_model(
        "MaxPool",
        {"x": x},
        12,
        outputs=("y", "indices"),
        kernel_shape=[2] * spatial,
        pads=[1] * spatial + [0] * spatial,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("dtype", [np.int8, np.uint8])
@pytest.mark.parametrize(
    "attributes",
    [{"auto_pad": "SAME_UPPER"}, {"pads": [1, 1, 0, 0]}],
)
def test_maxpool_integer_input_with_pads(dtype, attributes):
    """The pads were filled with -inf, which no integer tensor can hold."""
    x = np.array([[[[1, 2], [3, 4]]]], dtype=dtype)
    model = make_single_node_model(
        "MaxPool", {"x": x}, 12, kernel_shape=[2, 2], **attributes
    )
    assert_matches_oracle(model, {"x": x})
