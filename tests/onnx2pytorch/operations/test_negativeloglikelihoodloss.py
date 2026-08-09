import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_model(x, target, weight=None, **attrs):
    inputs = ["x", "target"]
    value_infos = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
        helper.make_tensor_value_info("target", TensorProto.INT64, list(target.shape)),
    ]
    feed = {"x": x, "target": target}
    if weight is not None:
        inputs.append("weight")
        value_infos.append(
            helper.make_tensor_value_info(
                "weight", TensorProto.FLOAT, list(weight.shape)
            )
        )
        feed["weight"] = weight

    node = helper.make_node(
        "NegativeLogLikelihoodLoss", inputs=inputs, outputs=["loss"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "nllloss_test",
        value_infos,
        [helper.make_tensor_value_info("loss", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model, feed


def check_nll_loss(x, target, weight=None, **attrs):
    model, feed = build_model(x, target, weight, **attrs)
    exp_loss = ort.InferenceSession(model.SerializeToString()).run(None, feed)[0]
    with torch.no_grad():
        loss = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    np.testing.assert_allclose(loss.numpy(), exp_loss, rtol=1e-5, atol=1e-6)


def make_log_probabilities(shape, seed):
    np.random.seed(seed)
    scores = np.random.randn(*shape).astype(np.float32)
    scores = scores - np.log(np.exp(scores).sum(axis=1, keepdims=True))
    return scores.astype(np.float32)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_nll_loss_reduction(reduction):
    x = make_log_probabilities((5, 4), 0)
    target = np.random.randint(0, 4, size=(5,)).astype(np.int64)
    check_nll_loss(x, target, reduction=reduction)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_nll_loss_weights(reduction):
    x = make_log_probabilities((6, 3), 1)
    target = np.random.randint(0, 3, size=(6,)).astype(np.int64)
    weight = np.array([0.2, 0.5, 1.5], dtype=np.float32)
    check_nll_loss(x, target, weight, reduction=reduction)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_nll_loss_ignore_index(reduction):
    x = make_log_probabilities((6, 3), 2)
    target = np.array([0, 1, 2, 1, 1, 0], dtype=np.int64)
    check_nll_loss(x, target, reduction=reduction, ignore_index=1)


def test_nll_loss_ignore_index_with_weights():
    x = make_log_probabilities((6, 3), 3)
    target = np.array([0, 1, 2, 1, 2, 0], dtype=np.int64)
    weight = np.array([0.3, 0.7, 2.0], dtype=np.float32)
    check_nll_loss(x, target, weight, reduction="mean", ignore_index=1)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_nll_loss_multidimensional(reduction):
    x = make_log_probabilities((3, 5, 2, 4), 4)
    target = np.random.randint(0, 5, size=(3, 2, 4)).astype(np.int64)
    check_nll_loss(x, target, reduction=reduction)


def test_nll_loss_multidimensional_weights_and_ignore_index():
    x = make_log_probabilities((2, 4, 3), 5)
    target = np.random.randint(0, 4, size=(2, 3)).astype(np.int64)
    weight = np.array([1.0, 0.4, 0.9, 2.2], dtype=np.float32)
    check_nll_loss(x, target, weight, reduction="mean", ignore_index=2)


def test_nll_loss_default_reduction_is_mean():
    x = make_log_probabilities((4, 3), 6)
    target = np.random.randint(0, 3, size=(4,)).astype(np.int64)
    check_nll_loss(x, target)
