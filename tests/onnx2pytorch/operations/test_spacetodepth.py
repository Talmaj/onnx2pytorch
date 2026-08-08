import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_space_to_depth(x, blocksize):
    node = helper.make_node(
        "SpaceToDepth", inputs=["x"], outputs=["y"], blocksize=blocksize
    )
    graph = helper.make_graph(
        [node],
        "spacetodepth_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp_y = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})[0]

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("blocksize", [2, 3])
def test_space_to_depth(blocksize):
    np.random.seed(0)
    x = np.random.randn(2, 3, 2 * blocksize, 3 * blocksize).astype(np.float32)
    check_space_to_depth(x, blocksize)


def test_space_to_depth_single_batch():
    np.random.seed(0)
    x = np.random.randn(1, 1, 4, 6).astype(np.float32)
    check_space_to_depth(x, 2)


def test_space_to_depth_roundtrip_with_depth_to_space():
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)

    nodes = [
        helper.make_node("SpaceToDepth", inputs=["x"], outputs=["t"], blocksize=2),
        helper.make_node(
            "DepthToSpace", inputs=["t"], outputs=["y"], blocksize=2, mode="DCR"
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "spacetodepth_roundtrip_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    with torch.no_grad():
        y = ConvertModel(model)(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), x, rtol=1e-5, atol=1e-5)
