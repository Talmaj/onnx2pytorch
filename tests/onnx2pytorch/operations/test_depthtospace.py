import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_depth_to_space(x, blocksize, mode=None):
    attrs = {"blocksize": blocksize}
    if mode is not None:
        attrs["mode"] = mode
    node = helper.make_node("DepthToSpace", inputs=["x"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "depthtospace_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("mode", [None, "DCR", "CRD"])
@pytest.mark.parametrize("blocksize", [2, 3])
def test_depth_to_space(mode, blocksize):
    np.random.seed(0)
    x = np.random.randn(2, 2 * blocksize**2, 3, 4).astype(np.float32)
    check_depth_to_space(x, blocksize, mode)


@pytest.mark.parametrize("mode", ["DCR", "CRD"])
def test_depth_to_space_single_channel(mode):
    np.random.seed(0)
    x = np.random.randn(1, 4, 2, 2).astype(np.float32)
    check_depth_to_space(x, 2, mode)
