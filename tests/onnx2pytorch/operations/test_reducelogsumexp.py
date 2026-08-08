import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel
from onnx2pytorch.operations.reducelogsumexp import ReduceLogSumExp


def check_reduce_log_sum_exp(x, axes, keepdims, opset_version=13):
    inputs = ["x"]
    initializers = []
    attrs = {"keepdims": keepdims}
    if axes is not None:
        if opset_version >= 18:
            inputs.append("axes")
            initializers.append(
                helper.make_tensor("axes", TensorProto.INT64, [len(axes)], axes)
            )
        else:
            attrs["axes"] = axes

    node = helper.make_node("ReduceLogSumExp", inputs=inputs, outputs=["y"], **attrs)
    graph = helper.make_graph(
        [node],
        "reducelogsumexp_test",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape))],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
        initializers,
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, {"x": x})[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(torch.from_numpy(x))

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("opset_version", [13, 18])
@pytest.mark.parametrize("keepdims", [0, 1])
@pytest.mark.parametrize("axes", [None, [0], [1], [-1], [0, 2]])
def test_reduce_log_sum_exp(opset_version, keepdims, axes):
    np.random.seed(0)
    x = np.random.randn(3, 4, 5).astype(np.float32)
    check_reduce_log_sum_exp(x, axes, keepdims, opset_version)


def test_reduce_log_sum_exp_noop_with_empty_axes():
    x = torch.randn(2, 3, 4)

    op = ReduceLogSumExp(noop_with_empty_axes=True)
    torch.testing.assert_close(op(x), x)

    op = ReduceLogSumExp()
    torch.testing.assert_close(op(x), torch.logsumexp(x.view(-1), 0).view(1, 1, 1))
