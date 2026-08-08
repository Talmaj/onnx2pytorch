import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_grid_sample(x, grid, opset_version=16, **attrs):
    node = helper.make_node("GridSample", inputs=["x", "grid"], outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "gridsample_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info("grid", TensorProto.FLOAT, list(grid.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x, "grid": grid})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x), torch.from_numpy(grid))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-5)


def make_inputs():
    np.random.seed(0)
    x = np.random.randn(2, 3, 4, 4).astype(np.float32)
    grid = np.random.uniform(-1.5, 1.5, size=(2, 5, 5, 2)).astype(np.float32)
    return x, grid


@pytest.mark.parametrize("mode", ["bilinear", "nearest", "bicubic"])
@pytest.mark.parametrize("padding_mode", ["zeros", "border", "reflection"])
@pytest.mark.parametrize("align_corners", [0, 1])
def test_grid_sample(mode, padding_mode, align_corners):
    x, grid = make_inputs()
    check_grid_sample(
        x,
        grid,
        mode=mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


@pytest.mark.parametrize("mode", ["linear", "nearest", "cubic"])
def test_grid_sample_opset20_mode_names(mode):
    x, grid = make_inputs()
    check_grid_sample(x, grid, opset_version=20, mode=mode)


def test_grid_sample_defaults():
    x, grid = make_inputs()
    check_grid_sample(x, grid)
