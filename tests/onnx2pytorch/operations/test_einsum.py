import numpy as np
import onnxruntime as ort
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_einsum(equation, *inputs):
    names = ["x{}".format(i) for i in range(len(inputs))]
    node = helper.make_node("Einsum", inputs=names, outputs=["y"], equation=equation)
    graph = helper.make_graph(
        [node],
        "einsum_test",
        [
            helper.make_tensor_value_info(name, TensorProto.FLOAT, list(x.shape))
            for name, x in zip(names, inputs)
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 12)])

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(None, dict(zip(names, inputs)))[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(*[torch.from_numpy(x) for x in inputs])

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-5, atol=1e-5)


def test_einsum_transpose():
    np.random.seed(0)
    check_einsum("ij->ji", np.random.randn(3, 4).astype(np.float32))


def test_einsum_sum_along_axis():
    np.random.seed(0)
    check_einsum("ij->i", np.random.randn(3, 4).astype(np.float32))


def test_einsum_inner_product():
    np.random.seed(0)
    check_einsum(
        "i,i",
        np.random.randn(5).astype(np.float32),
        np.random.randn(5).astype(np.float32),
    )


def test_einsum_matmul():
    np.random.seed(0)
    check_einsum(
        "ij,jk->ik",
        np.random.randn(3, 4).astype(np.float32),
        np.random.randn(4, 5).astype(np.float32),
    )


def test_einsum_batch_matmul():
    np.random.seed(0)
    check_einsum(
        "bij,bjk->bik",
        np.random.randn(2, 3, 4).astype(np.float32),
        np.random.randn(2, 4, 5).astype(np.float32),
    )
