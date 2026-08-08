import numpy as np
import torch
from onnx import helper, TensorProto
from onnx.reference import ReferenceEvaluator

from onnx2pytorch.convert import ConvertModel


def check_deform_conv(x, w, offset, b=None, mask=None, **attrs):
    inputs = ["x", "w", "offset"]
    value_infos = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
        helper.make_tensor_value_info("w", TensorProto.FLOAT, list(w.shape)),
        helper.make_tensor_value_info("offset", TensorProto.FLOAT, list(offset.shape)),
    ]
    feed = {"x": x, "w": w, "offset": offset}
    if b is not None or mask is not None:
        inputs.append("b" if b is not None else "")
        if b is not None:
            value_infos.append(
                helper.make_tensor_value_info("b", TensorProto.FLOAT, list(b.shape))
            )
            feed["b"] = b
    if mask is not None:
        inputs.append("mask")
        value_infos.append(
            helper.make_tensor_value_info("mask", TensorProto.FLOAT, list(mask.shape))
        )
        feed["mask"] = mask

    node = helper.make_node("DeformConv", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "deformconv_test",
        value_infos,
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 19)])

    # onnxruntime has no DeformConv kernel, compare against the onnx reference
    exp_y = ReferenceEvaluator(model).run(None, feed)[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(*[torch.from_numpy(v) for v in feed.values()])

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-5)


def make_offset(n, offset_group, kernel, out_shape, seed=1):
    rng = np.random.RandomState(seed)
    shape = (n, 2 * offset_group * kernel[0] * kernel[1]) + out_shape
    return rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)


def test_deform_conv():
    np.random.seed(0)
    x = np.random.randn(1, 1, 4, 4).astype(np.float32)
    w = np.random.randn(1, 1, 2, 2).astype(np.float32)
    offset = make_offset(1, 1, (2, 2), (3, 3))
    check_deform_conv(x, w, offset, kernel_shape=[2, 2])


def test_deform_conv_with_bias_and_mask():
    np.random.seed(0)
    x = np.random.randn(2, 2, 5, 5).astype(np.float32)
    w = np.random.randn(3, 2, 3, 3).astype(np.float32)
    b = np.random.randn(3).astype(np.float32)
    offset = make_offset(2, 1, (3, 3), (3, 3))
    mask = np.random.rand(2, 1 * 3 * 3, 3, 3).astype(np.float32)
    check_deform_conv(x, w, offset, b=b, mask=mask, kernel_shape=[3, 3])


def test_deform_conv_pads_strides_dilations():
    np.random.seed(0)
    x = np.random.randn(1, 1, 6, 6).astype(np.float32)
    w = np.random.randn(2, 1, 2, 2).astype(np.float32)
    offset = make_offset(1, 1, (2, 2), (3, 3))
    check_deform_conv(
        x,
        w,
        offset,
        kernel_shape=[2, 2],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[2, 2],
    )


def test_deform_conv_groups():
    np.random.seed(0)
    x = np.random.randn(1, 4, 5, 5).astype(np.float32)
    w = np.random.randn(4, 2, 2, 2).astype(np.float32)
    offset = make_offset(1, 2, (2, 2), (4, 4))
    check_deform_conv(x, w, offset, kernel_shape=[2, 2], group=2, offset_group=2)
