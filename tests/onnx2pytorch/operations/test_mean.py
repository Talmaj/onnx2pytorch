import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_mean(*inputs):
    names = ["x{}".format(i) for i in range(len(inputs))]
    node = helper.make_node("Mean", inputs=names, outputs=["y"])
    graph = helper.make_graph(
        [node],
        "mean_test",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(x.shape))
            for name, x in zip(names, inputs)
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, dict(zip(names, inputs)))[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(*[torch.from_numpy(x) for x in inputs])

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-6, atol=1e-6)


def test_mean_two_inputs():
    check_mean(
        np.array([3.0, 0.0, 2.0], dtype=np.float32),
        np.array([1.0, 3.0, 4.0], dtype=np.float32),
    )


def test_mean_three_inputs():
    check_mean(
        np.array([3.0, 0.0, 2.0], dtype=np.float32),
        np.array([1.0, 3.0, 4.0], dtype=np.float32),
        np.array([2.0, 6.0, 6.0], dtype=np.float32),
    )


def test_mean_single_input():
    check_mean(np.array([3.0, 0.0, 2.0], dtype=np.float32))


def test_mean_broadcast():
    np.random.seed(0)
    check_mean(
        np.random.randn(2, 3, 4).astype(np.float32),
        np.random.randn(4).astype(np.float32),
        np.random.randn(3, 1).astype(np.float32),
    )
