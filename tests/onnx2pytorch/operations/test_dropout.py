import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_dropout(x, outputs, ratio=None, training_mode=None):
    inputs = ["x"]
    initializers = []
    if ratio is not None:
        inputs.append("ratio")
        initializers.append(helper.make_tensor("ratio", TensorProto.FLOAT, [], [ratio]))
    if training_mode is not None:
        if ratio is None:
            inputs.append("")
        inputs.append("training_mode")
        initializers.append(
            helper.make_tensor("training_mode", TensorProto.BOOL, [], [training_mode])
        )

    node = helper.make_node("Dropout", inputs=inputs, outputs=outputs)
    graph_outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    if len(outputs) == 2:
        graph_outputs.append(
            helper.make_tensor_value_info("mask", TensorProto.BOOL, None)
        )
    graph = helper.make_graph(
        [node],
        "dropout_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        graph_outputs,
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    exp = ort.InferenceSession(model.SerializeToString()).run(None, {"x": x})

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        res = o2p_model(torch.from_numpy(x))
    if len(outputs) == 1:
        res = [res]

    for actual, expected in zip(res, exp):
        np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-5)


def test_dropout_single_output():
    np.random.seed(0)
    check_dropout(np.random.randn(3, 4).astype(np.float32), ["y"])


def test_dropout_with_mask():
    np.random.seed(0)
    check_dropout(np.random.randn(3, 4, 5).astype(np.float32), ["y", "mask"])


@pytest.mark.parametrize("ratio", [0.0, 0.5])
def test_dropout_with_ratio(ratio):
    np.random.seed(0)
    check_dropout(np.random.randn(2, 3).astype(np.float32), ["y", "mask"], ratio=ratio)


def test_dropout_training_mode_false():
    np.random.seed(0)
    check_dropout(
        np.random.randn(2, 3).astype(np.float32),
        ["y", "mask"],
        ratio=0.5,
        training_mode=False,
    )
