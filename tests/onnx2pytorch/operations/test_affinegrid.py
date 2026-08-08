import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_affine_grid(theta, size, **attrs):
    node = helper.make_node(
        "AffineGrid", inputs=["theta", "size"], outputs=["grid"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "affinegrid_test",
        [
            helper.make_tensor_value_info(
                "theta", TensorProto.FLOAT, list(theta.shape)
            ),
            helper.make_tensor_value_info("size", TensorProto.INT64, list(size.shape)),
        ],
        [helper.make_tensor_value_info("grid", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])

    exp_grid = ort.InferenceSession(model.SerializeToString()).run(
        None, {"theta": theta, "size": size}
    )[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        grid = o2p_model(torch.from_numpy(theta), torch.from_numpy(size))

    np.testing.assert_allclose(grid.numpy(), exp_grid, rtol=1e-5, atol=1e-5)


def make_theta_2d(n=2):
    np.random.seed(0)
    angles = np.random.uniform(-1.0, 1.0, size=n)
    theta = np.zeros((n, 2, 3), dtype=np.float32)
    for i, angle in enumerate(angles):
        theta[i] = [
            [np.cos(angle), -np.sin(angle), 0.3],
            [np.sin(angle), np.cos(angle), -0.2],
        ]
    return theta


def make_theta_3d(n=2):
    np.random.seed(1)
    return np.random.uniform(-1.0, 1.0, size=(n, 3, 4)).astype(np.float32)


@pytest.mark.parametrize("align_corners", [0, 1])
def test_affine_grid_2d(align_corners):
    theta = make_theta_2d()
    size = np.array([2, 3, 5, 6], dtype=np.int64)
    check_affine_grid(theta, size, align_corners=align_corners)


@pytest.mark.parametrize("align_corners", [0, 1])
def test_affine_grid_3d(align_corners):
    theta = make_theta_3d()
    size = np.array([2, 3, 4, 5, 6], dtype=np.int64)
    check_affine_grid(theta, size, align_corners=align_corners)


def test_affine_grid_default_align_corners():
    theta = make_theta_2d(n=1)
    size = np.array([1, 1, 4, 4], dtype=np.int64)
    check_affine_grid(theta, size)


def test_affine_grid_non_square():
    theta = make_theta_2d(n=3)
    size = np.array([3, 2, 2, 7], dtype=np.int64)
    check_affine_grid(theta, size, align_corners=1)
