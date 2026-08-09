import numpy as np
import pytest
import torch
from onnx import defs, helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.swiglu import SwiGLU


def convert_swiglu_model(a, b, **attrs):
    node = helper.make_node("SwiGLU", inputs=["a", "b"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "swiglu_test",
        [
            helper.make_tensor_value_info("a", TensorProto.FLOAT, list(a.shape)),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(a.shape))],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", defs.onnx_opset_version())]
    )
    return ConvertModel(model)


def test_swiglu_still_has_no_oracle():
    """Once onnx defines SwiGLU these tests should compare against a runtime."""
    names = {schema.name for schema in defs.get_all_schemas_with_history()}
    assert "SwiGLU" not in names


@pytest.mark.parametrize("alpha", [None, 1.0, 0.5, 2.0])
def test_swiglu(alpha):
    np.random.seed(0)
    a = np.random.randn(3, 4, 5).astype(np.float32)
    b = np.random.randn(3, 4, 5).astype(np.float32)

    attrs = {} if alpha is None else {"alpha": alpha}
    o2p_model = convert_swiglu_model(a, b, **attrs)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(a), torch.from_numpy(b))

    x = 1.0 if alpha is None else alpha
    exp_y = a / (1 + np.exp(-x * a)) * b
    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_swiglu_gate_is_swish():
    a = torch.randn(3, 4)
    b = torch.ones(3, 4)

    op = SwiGLU()
    torch.testing.assert_close(op(a, b), torch.nn.functional.silu(a))
