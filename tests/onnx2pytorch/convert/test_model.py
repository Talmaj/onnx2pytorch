import numpy as np
import onnx
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.convert.model import get_opset_version


def make_relu_model(opset_imports):
    node = helper.make_node("Relu", ["x"], ["y"])
    graph = helper.make_graph(
        [node],
        "relu_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])],
    )
    return helper.make_model(graph, opset_imports=opset_imports)


def test_get_opset_version_default_domain_not_first():
    model = make_relu_model(
        [helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 17)]
    )
    assert get_opset_version(model) == 17


def test_get_opset_version_ai_onnx_alias():
    model = make_relu_model([helper.make_opsetid("ai.onnx", 13)])
    assert get_opset_version(model) == 13


def test_get_opset_version_warns_about_unknown_domains():
    model = make_relu_model(
        [helper.make_opsetid("", 17), helper.make_opsetid("com.microsoft", 1)]
    )
    with pytest.warns(UserWarning, match="com.microsoft"):
        assert get_opset_version(model) == 17


def test_get_opset_version_without_default_domain():
    model = make_relu_model([helper.make_opsetid("com.microsoft", 1)])
    with pytest.raises(ValueError, match="com.microsoft"):
        get_opset_version(model)


def test_get_opset_version_without_any_import():
    model = make_relu_model([])
    with pytest.raises(ValueError, match="none"):
        get_opset_version(model)


def test_convert_model_picks_default_domain_version():
    """A non-default domain must not decide which opset the converter assumes."""
    model = make_relu_model(
        [helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 17)]
    )
    converted = ConvertModel(model)
    x = np.array([[-1.0, 0.0, 1.0], [2.0, -3.0, 4.0]], dtype=np.float32)
    with torch.no_grad():
        y = converted(torch.from_numpy(x))
    np.testing.assert_array_equal(y.numpy(), np.maximum(x, 0))
