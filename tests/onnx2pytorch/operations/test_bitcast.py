import numpy as np
import pytest
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.bitcast import BitCast


def build_model(x, in_type, to_type):
    node = helper.make_node("BitCast", inputs=["x"], outputs=["y"], to=to_type)
    graph = helper.make_graph(
        [node],
        "bitcast_test",
        [helper.make_tensor_value_info("x", in_type, list(x.shape))],
        [helper.make_tensor_value_info("y", to_type, list(x.shape))],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 26)])


@pytest.mark.parametrize(
    "np_dtype, onnx_type, to_onnx_type",
    [
        (np.float32, TensorProto.FLOAT, TensorProto.INT32),
        (np.int32, TensorProto.INT32, TensorProto.FLOAT),
        (np.float64, TensorProto.DOUBLE, TensorProto.INT64),
        (np.int8, TensorProto.INT8, TensorProto.UINT8),
    ],
)
def test_bitcast(np_dtype, onnx_type, to_onnx_type):
    np.random.seed(0)
    x = (np.random.randn(3, 4) * 100).astype(np_dtype)
    model = build_model(x, onnx_type, to_onnx_type)

    # onnxruntime 1.28 has no BitCast kernel
    exp_y = ReferenceEvaluator(model).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_array_equal(y.numpy(), exp_y)


def test_bitcast_preserves_bits():
    x = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)

    op = BitCast(dtype="int32")
    assert torch.equal(op(x), torch.tensor([1065353216, -1082130432, 0]))

    back = BitCast(dtype="float")
    assert torch.equal(back(op(x)), x)
