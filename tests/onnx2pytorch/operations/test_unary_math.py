import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_unary_op(op_type, x, opset_version=13):
    node = helper.make_node(op_type, inputs=["x"], outputs=["y"])
    graph = helper.make_graph(
        [node],
        "{}_test".format(op_type.lower()),
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(x.shape))],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_abs():
    check_unary_op("Abs", np.array([-1.5, 0.0, 1.5], dtype=np.float32))
    np.random.seed(0)
    check_unary_op("Abs", np.random.randn(3, 4, 5).astype(np.float32))
