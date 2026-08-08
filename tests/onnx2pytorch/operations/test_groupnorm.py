import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_group_normalization(x, scale, bias, num_groups, epsilon, opset_version):
    node = helper.make_node(
        "GroupNormalization",
        inputs=["x", "scale", "bias"],
        outputs=["y"],
        num_groups=num_groups,
        epsilon=epsilon,
    )
    graph = helper.make_graph(
        [node],
        "groupnormalization_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info(
                "scale", TensorProto.FLOAT, list(scale.shape)
            ),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, list(bias.shape)),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x, "scale": scale, "bias": bias})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(
            torch.from_numpy(x), torch.from_numpy(scale), torch.from_numpy(bias)
        )

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("num_groups", [1, 2, 4])
@pytest.mark.parametrize("epsilon", [1e-5, 1e-2])
def test_group_normalization_opset21(num_groups, epsilon):
    np.random.seed(0)
    x = np.random.randn(2, 4, 3, 3).astype(np.float32)
    scale = np.random.randn(4).astype(np.float32)
    bias = np.random.randn(4).astype(np.float32)
    check_group_normalization(x, scale, bias, num_groups, epsilon, 21)


@pytest.mark.parametrize("num_groups", [1, 2, 4])
def test_group_normalization_opset18(num_groups):
    np.random.seed(0)
    x = np.random.randn(2, 4, 3, 3).astype(np.float32)
    scale = np.random.randn(num_groups).astype(np.float32)
    bias = np.random.randn(num_groups).astype(np.float32)
    check_group_normalization(x, scale, bias, num_groups, 1e-5, 18)


def test_group_normalization_3d_input():
    np.random.seed(0)
    x = np.random.randn(2, 6, 5).astype(np.float32)
    scale = np.random.randn(6).astype(np.float32)
    bias = np.random.randn(6).astype(np.float32)
    check_group_normalization(x, scale, bias, 3, 1e-5, 21)
