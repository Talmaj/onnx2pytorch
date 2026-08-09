import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def build_model(scores, labels, weights=None, log_prob=False, **attrs):
    inputs = ["scores", "labels"]
    value_infos = [
        helper.make_tensor_value_info("scores", TensorProto.FLOAT, list(scores.shape)),
        helper.make_tensor_value_info("labels", TensorProto.INT64, list(labels.shape)),
    ]
    feed = {"scores": scores, "labels": labels}
    if weights is not None:
        inputs.append("weights")
        value_infos.append(
            helper.make_tensor_value_info(
                "weights", TensorProto.FLOAT, list(weights.shape)
            )
        )
        feed["weights"] = weights

    outputs = ["loss"] + (["log_prob"] if log_prob else [])
    node = helper.make_node(
        "SoftmaxCrossEntropyLoss", inputs=inputs, outputs=outputs, **attrs
    )
    graph = helper.make_graph(
        [node],
        "softmaxcrossentropyloss_test",
        value_infos,
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
            for name in outputs
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    return model, feed


def check_loss(scores, labels, weights=None, log_prob=False, **attrs):
    model, feed = build_model(scores, labels, weights, log_prob, **attrs)
    expected = ort.InferenceSession(model.SerializeToString()).run(None, feed)
    with torch.no_grad():
        out = ConvertModel(model)(*[torch.from_numpy(v) for v in feed.values()])
    if not log_prob:
        out = [out]
    for actual, exp in zip(out, expected):
        np.testing.assert_allclose(actual.numpy(), exp, rtol=1e-5, atol=1e-6)


def make_scores(shape, seed):
    np.random.seed(seed)
    return np.random.randn(*shape).astype(np.float32)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_softmax_cross_entropy_loss_reduction(reduction):
    scores = make_scores((5, 4), 0)
    labels = np.random.randint(0, 4, size=(5,)).astype(np.int64)
    check_loss(scores, labels, reduction=reduction)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_softmax_cross_entropy_loss_weights(reduction):
    scores = make_scores((6, 3), 1)
    labels = np.random.randint(0, 3, size=(6,)).astype(np.int64)
    weights = np.array([0.2, 0.5, 1.5], dtype=np.float32)
    check_loss(scores, labels, weights, reduction=reduction)


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_softmax_cross_entropy_loss_ignore_index(reduction):
    scores = make_scores((6, 3), 2)
    labels = np.array([0, 1, 2, 1, 1, 0], dtype=np.int64)
    check_loss(scores, labels, reduction=reduction, ignore_index=1)


def test_softmax_cross_entropy_loss_ignore_index_with_weights():
    scores = make_scores((6, 3), 3)
    labels = np.array([0, 1, 2, 1, 2, 0], dtype=np.int64)
    weights = np.array([0.3, 0.7, 2.0], dtype=np.float32)
    check_loss(scores, labels, weights, reduction="mean", ignore_index=1)


def test_softmax_cross_entropy_loss_log_prob():
    scores = make_scores((4, 5), 4)
    labels = np.random.randint(0, 5, size=(4,)).astype(np.int64)
    check_loss(scores, labels, log_prob=True, reduction="mean")


def test_softmax_cross_entropy_loss_log_prob_none_reduction():
    scores = make_scores((3, 4), 5)
    labels = np.random.randint(0, 4, size=(3,)).astype(np.int64)
    check_loss(scores, labels, log_prob=True, reduction="none")


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_softmax_cross_entropy_loss_multidimensional(reduction):
    scores = make_scores((3, 5, 2, 4), 6)
    labels = np.random.randint(0, 5, size=(3, 2, 4)).astype(np.int64)
    check_loss(scores, labels, log_prob=True, reduction=reduction)


def test_softmax_cross_entropy_loss_default_reduction_is_mean():
    scores = make_scores((4, 3), 7)
    labels = np.random.randint(0, 3, size=(4,)).astype(np.int64)
    check_loss(scores, labels)
