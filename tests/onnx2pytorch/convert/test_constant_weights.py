"""Weights produced by Constant nodes, not only graph initializers."""

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnx import TensorProto, helper, numpy_helper
from torch.nn.modules.conv import _ConvNd

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.batchnorm import BatchNormWrapper
from onnx2pytorch.operations.dynamicconv import DynamicConv

RTOL = 1e-4
ATOL = 1e-5


def _model(nodes, inputs, outputs, initializers=(), opset=14):
    graph = helper.make_graph(
        nodes, "g", inputs, outputs, initializer=list(initializers)
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 8
    return model


def _vi(name, shape, dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, shape)


def _constant(name, array):
    return helper.make_node(
        "Constant",
        [],
        [name],
        value=numpy_helper.from_array(np.ascontiguousarray(array), name),
    )


def _check(model, feeds):
    payload = model.SerializeToString()
    expected = ort.InferenceSession(payload, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    converted = ConvertModel(onnx.load_from_string(payload))
    converted.eval()
    names = [value.name for value in onnx.load_from_string(payload).graph.input]
    with torch.no_grad():
        actual = converted(
            *[torch.from_numpy(np.asarray(feeds[name])) for name in names]
        )
    if not isinstance(actual, (list, tuple)):
        actual = (actual,)
    for got, want in zip(actual, expected):
        got = got.detach().cpu().numpy()
        np.testing.assert_allclose(got, want, rtol=RTOL, atol=ATOL)
    return converted


def _has(converted, cls):
    return any(isinstance(module, cls) for module in converted.modules())


def test_conv_weights_from_constant_nodes():
    rng = np.random.default_rng(0)
    cases = {
        "conv1d": (
            "Conv",
            rng.standard_normal((2, 3, 8)).astype(np.float32),
            rng.standard_normal((4, 3, 3)).astype(np.float32),
            rng.standard_normal((4,)).astype(np.float32),
            [3],
        ),
        "conv2d": (
            "Conv",
            rng.standard_normal((2, 3, 8, 8)).astype(np.float32),
            rng.standard_normal((4, 3, 3, 3)).astype(np.float32),
            rng.standard_normal((4,)).astype(np.float32),
            [3, 3],
        ),
        "convtranspose2d": (
            "ConvTranspose",
            rng.standard_normal((2, 3, 8, 8)).astype(np.float32),
            rng.standard_normal((3, 4, 3, 3)).astype(np.float32),
            rng.standard_normal((4,)).astype(np.float32),
            [3, 3],
        ),
    }
    for _name, (op_type, x, weight, bias, kernel) in cases.items():
        nodes = [
            _constant("W0", weight),
            helper.make_node("Identity", ["W0"], ["W"]),
            _constant("B", bias),
            helper.make_node(op_type, ["X", "W", "B"], ["Y"], kernel_shape=kernel),
        ]
        model = _model([*nodes], [_vi("X", list(x.shape))], [_vi("Y", None)])
        converted = _check(model, {"X": x})
        assert _has(converted, _ConvNd)
        assert not _has(converted, DynamicConv)


def test_conv_inside_if_branch_and_outer_constant():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 2, 5, 5)).astype(np.float32)
    weight = rng.standard_normal((3, 2, 1, 1)).astype(np.float32)
    other = rng.standard_normal((3, 2, 1, 1)).astype(np.float32)
    bias = rng.standard_normal((3,)).astype(np.float32)

    def branch(weight_nodes, weight_name, output_name):
        return helper.make_graph(
            [
                *weight_nodes,
                helper.make_node(
                    "Conv",
                    ["X", weight_name, "B"],
                    [output_name],
                    kernel_shape=[1, 1],
                ),
            ],
            output_name,
            [],
            [_vi(output_name, None)],
        )

    then_branch = branch([_constant("Wt", weight)], "Wt", "Yt")
    else_branch = branch([], "Wouter", "Ye")
    nodes = [
        _constant("Wouter0", other),
        helper.make_node("Identity", ["Wouter0"], ["Wouter"]),
        _constant("B", bias),
        helper.make_node(
            "If",
            ["cond"],
            ["Y"],
            then_branch=then_branch,
            else_branch=else_branch,
        ),
    ]
    model = _model(
        nodes,
        [_vi("X", list(x.shape)), _vi("cond", [], TensorProto.BOOL)],
        [_vi("Y", None)],
    )
    true_model = _check(model, {"X": x, "cond": np.array(True)})
    false_model = _check(model, {"X": x, "cond": np.array(False)})
    assert _has(true_model, _ConvNd)
    assert _has(false_model, _ConvNd)


def test_dynamic_conv_weight_is_an_input():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((1, 3, 6, 6)).astype(np.float32)
    weight = rng.standard_normal((4, 3, 3, 3)).astype(np.float32)
    bias = rng.standard_normal((4,)).astype(np.float32)
    node = helper.make_node("Conv", ["X", "W", "B"], ["Y"], kernel_shape=[3, 3])
    model = _model(
        [node],
        [
            _vi("X", list(x.shape)),
            _vi("W", list(weight.shape)),
            _vi("B", list(bias.shape)),
        ],
        [_vi("Y", None)],
    )
    converted = _check(model, {"X": x, "W": weight, "B": bias})
    assert _has(converted, DynamicConv)


def test_scalar_constant_keeps_its_rank():
    """A folded 0-d index must stay scalar, or Gather keeps a dimension.

    numpy_helper.from_array turns a 0-d array into shape (1,), so the Constant
    is built the way Silero does: an empty ``dims`` field.
    """
    rng = np.random.default_rng(5)
    data = rng.standard_normal((2, 1, 4)).astype(np.float32)
    index = numpy_helper.from_array(np.array([1], dtype=np.int64), "index")
    del index.dims[:]
    model = _model(
        [
            helper.make_node("Constant", [], ["index"], value=index),
            helper.make_node("Gather", ["data", "index"], ["Y"], axis=0),
        ],
        [_vi("data", list(data.shape))],
        [_vi("Y", None)],
    )
    converted = _check(model, {"data": data})
    with torch.no_grad():
        out = converted(torch.from_numpy(data))
    assert tuple(out.shape) == (1, 4)


def test_dynamic_gemm_weight_is_an_input():
    rng = np.random.default_rng(4)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    c = rng.standard_normal((5,)).astype(np.float32)
    model = _model(
        [
            helper.make_node(
                "Gemm", ["A", "B", "C"], ["Y"], alpha=0.5, beta=1.5, transB=1
            )
        ],
        [_vi("A", list(a.shape)), _vi("B", [5, 3]), _vi("C", list(c.shape))],
        [_vi("Y", None)],
    )
    b_t = rng.standard_normal((5, 3)).astype(np.float32)
    _check(model, {"A": a, "B": b_t, "C": c})


def test_gemm_and_batchnorm_weights_from_constants():
    rng = np.random.default_rng(3)
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((4, 3)).astype(np.float32)
    c = rng.standard_normal((4,)).astype(np.float32)
    gemm = _model(
        [
            _constant("B", b),
            _constant("C", c),
            helper.make_node("Gemm", ["A", "B", "C"], ["Y"], transB=1),
        ],
        [_vi("A", list(a.shape))],
        [_vi("Y", None)],
    )
    converted = _check(gemm, {"A": a})
    assert any(isinstance(module, torch.nn.Linear) for module in converted.modules())

    x = rng.standard_normal((2, 3, 4, 4)).astype(np.float32)
    scale = rng.random(3).astype(np.float32) + 0.5
    shift = rng.standard_normal(3).astype(np.float32)
    mean = rng.standard_normal(3).astype(np.float32)
    var = rng.random(3).astype(np.float32) + 0.5
    batchnorm = _model(
        [
            _constant("scale", scale),
            _constant("shift", shift),
            _constant("mean", mean),
            _constant("var", var),
            helper.make_node(
                "BatchNormalization",
                ["X", "scale", "shift", "mean", "var"],
                ["Y"],
                epsilon=1e-5,
            ),
        ],
        [_vi("X", list(x.shape))],
        [_vi("Y", None)],
    )
    converted = _check(batchnorm, {"X": x})
    assert _has(converted, BatchNormWrapper)
