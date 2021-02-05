from unittest import TestCase

import pytest
import onnx
import numpy as np

from onnx2pytorch.convert import extract_attr_values, extract_attributes


@pytest.mark.parametrize(
    "kwargs, value",
    [
        [dict(type="INT", i=1), 1],
        [dict(type="FLOAT", f=np.float64(2.5)), 2.5],
        [dict(type="INTS", ints=(1, 2)), (1, 2)],
        [dict(type="FLOATS", floats=np.array((1.5, 2.5))), (1.5, 2.5)],
        [dict(type="STRING", s="nearest".encode()), "nearest"],
    ],
)
def test_extract_attr_values(kwargs, value):
    attr = onnx.AttributeProto(**kwargs)
    assert extract_attr_values(attr) == value


@pytest.mark.parametrize(
    "node, exp_kwargs",
    [
        [
            onnx.helper.make_node(
                "Conv",
                inputs=["x", "W"],
                outputs=["y"],
                kernel_shape=[3, 3],
                strides=[1, 1],
                dilations=[1, 1],
                group=1,
                pads=[1, 1, 1, 1],
            ),
            dict(
                kernel_size=(3, 3),
                stride=(1, 1),
                dilation=(1, 1),
                groups=1,
                padding=(1, 1),
            ),
        ],
        [
            onnx.helper.make_node(
                "Pad",
                inputs=["x", "pads", "value"],
                outputs=["y"],
                mode="constant",
                pads=[1, 0, 0, 1, 0, 0],
            ),
            dict(
                mode="constant",
                padding=[0, 0, 0, 0, 1, 1],
            ),
        ],
        [
            onnx.helper.make_node(
                "Flatten",
                inputs=["a"],
                outputs=["b"],
                axis=1,
            ),
            dict(
                start_dim=1,
            ),
        ],
        [
            onnx.helper.make_node(
                "Slice",
                inputs=["x", "starts", "ends", "axes", "steps"],
                outputs=["y"],
                starts=[0, 0, 3],
                ends=[20, 10, 4],
            ),
            dict(starts=(0, 0, 3), ends=(20, 10, 4)),
        ],
        [
            onnx.helper.make_node(
                "Resize",
                inputs=["X", "", "scales"],
                outputs=["Y"],
                mode="nearest",
                coordinate_transformation_mode="align_corners",
                extrapolation_value=1,
            ),
            dict(mode="nearest", align_corners=True, extrapolation_value=1),
        ],
        [
            onnx.helper.make_node(
                "AveragePool",
                inputs=["x"],
                outputs=["y"],
                kernel_shape=[3, 3],
                strides=[2, 2],
                ceil_mode=True,
                auto_pad="NOTSET",
            ),
            dict(kernel_size=(3, 3), stride=(2, 2), ceil_mode=True),
        ],
        [
            onnx.helper.make_node("LeakyRelu", inputs=["x"], outputs=["y"], alpha=0.5),
            dict(negative_slope=0.5),
        ],
    ],
)
def test_extract_attributes(node, exp_kwargs):
    extracted_kwargs = extract_attributes(node)
    TestCase().assertDictEqual(exp_kwargs, extracted_kwargs)
