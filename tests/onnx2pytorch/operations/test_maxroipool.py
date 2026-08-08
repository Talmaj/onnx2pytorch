import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_max_roi_pool(x, rois, **attrs):
    node = helper.make_node("MaxRoiPool", inputs=["x", "rois"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "maxroipool_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(rois.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x, "rois": rois})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x), torch.from_numpy(rois))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def make_inputs():
    np.random.seed(0)
    x = np.random.randn(2, 3, 8, 8).astype(np.float32)
    rois = np.array(
        [[0.0, 0.0, 0.0, 4.0, 4.0], [1.0, 2.0, 1.0, 7.0, 6.0]], dtype=np.float32
    )
    return x, rois


@pytest.mark.parametrize("pooled_shape", [[1, 1], [2, 2], [3, 2]])
def test_max_roi_pool(pooled_shape):
    x, rois = make_inputs()
    check_max_roi_pool(x, rois, pooled_shape=pooled_shape)


def test_max_roi_pool_spatial_scale():
    x, rois = make_inputs()
    check_max_roi_pool(x, rois, pooled_shape=[2, 2], spatial_scale=0.5)
