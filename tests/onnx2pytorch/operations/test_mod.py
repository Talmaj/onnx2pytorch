import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.mod import Mod


def check_mod(x, y, fmod, tensor_type):
    node = helper.make_node("Mod", inputs=["x", "y"], outputs=["z"], fmod=fmod)
    graph = helper.make_graph(
        [node],
        "mod_test",
        [
            helper.make_tensor_value_info("x", tensor_type, list(x.shape)),
            helper.make_tensor_value_info("y", tensor_type, list(y.shape)),
        ],
        [helper.make_tensor_value_info("z", tensor_type, list(x.shape))],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_z = ort_session.run(None, {"x": x, "y": y})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        z = o2p_model(torch.from_numpy(x), torch.from_numpy(y))

    np.testing.assert_allclose(z.numpy(), exp_z, rtol=1e-6, atol=1e-6)


def test_mod_int64():
    x = np.array([-4, 7, 5, 4, -7, 8], dtype=np.int64)
    y = np.array([2, -3, 8, -2, 3, 5], dtype=np.int64)
    check_mod(x, y, 0, TensorProto.INT64)
    check_mod(x, y, 1, TensorProto.INT64)


def test_mod_broadcast():
    np.random.seed(0)
    x = np.arange(0, 30).reshape([3, 2, 5]).astype(np.int32)
    y = np.array([7], dtype=np.int32)
    check_mod(x, y, 0, TensorProto.INT32)


def test_mod_float_fmod():
    x = np.array([-4.3, 7.2, 5.0, 4.3, -7.2, 8.0], dtype=np.float32)
    y = np.array([2.1, -3.4, 8.0, -2.1, 3.4, 5.0], dtype=np.float32)
    check_mod(x, y, 1, TensorProto.FLOAT)


@pytest.mark.parametrize("fmod", [0, 1])
def test_mod_sign(fmod):
    x = torch.tensor([-4, 7, 5, 4, -7, 8])
    y = torch.tensor([2, -3, 8, -2, 3, 5])

    op = Mod(fmod=fmod)
    if fmod:
        # Result has the sign of the dividend
        exp_z = torch.tensor([0, 1, 5, 0, -1, 3])
    else:
        # Result has the sign of the divisor
        exp_z = torch.tensor([0, -2, 5, 0, 2, 3])
    assert torch.equal(op(x, y), exp_z)
