import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_rms_normalization(x, scale, axis, epsilon):
    node = helper.make_node(
        "RMSNormalization",
        inputs=["x", "scale"],
        outputs=["y"],
        axis=axis,
        epsilon=epsilon,
    )
    graph = helper.make_graph(
        [node],
        "rmsnormalization_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info(
                "scale", TensorProto.FLOAT, list(scale.shape)
            ),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 23)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x, "scale": scale})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x), torch.from_numpy(scale))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("epsilon", [1e-5, 1e-2])
@pytest.mark.parametrize("axis", [-1, 2])
def test_rms_normalization_last_axis(axis, epsilon):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    check_rms_normalization(x, scale, axis, epsilon)


@pytest.mark.parametrize("axis", [-2, 1])
def test_rms_normalization_multiple_axes(axis):
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(3, 4).astype(np.float32)
    check_rms_normalization(x, scale, axis, 1e-5)


def test_rms_normalization_all_axes():
    np.random.seed(0)
    x = np.random.randn(2, 3, 4).astype(np.float32)
    scale = np.random.randn(2, 3, 4).astype(np.float32)
    check_rms_normalization(x, scale, 0, 1e-5)
