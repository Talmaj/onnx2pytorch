import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_mean_variance_normalization(x, axes):
    attrs = {} if axes is None else {"axes": axes}
    node = helper.make_node(
        "MeanVarianceNormalization", inputs=["x"], outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "meanvariancenormalization_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize(
    "axes", [None, [0], [0, 1], [0, 2, 3], [1, 2, 3], [0, 1, 2, 3]]
)
def test_mean_variance_normalization(axes):
    np.random.seed(0)
    x = np.random.randn(3, 4, 2, 2).astype(np.float32)
    check_mean_variance_normalization(x, axes)
