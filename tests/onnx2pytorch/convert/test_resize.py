import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from onnx2pytorch.convert import ConvertModel


def make_resize_model(node_inputs, initializers, out_shape):
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [1, 1, 2, 2])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, out_shape)

    # roi and scales are optional inputs and are omitted with an empty input name
    resize_node = onnx.helper.make_node(
        "Resize", inputs=node_inputs, outputs=["y"], mode="nearest"
    )

    graph = onnx.helper.make_graph([resize_node], "resize", [x], [y], initializers)
    model = onnx.helper.make_model_gen_version(
        graph,
        producer_name="resize-test",
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize("with_roi", [False, True])
def test_convert_resize_sizes_with_omitted_inputs(with_roi):
    sizes = onnx.helper.make_tensor("sizes", onnx.TensorProto.INT64, [4], [1, 1, 4, 4])
    initializers = [sizes]
    roi_name = ""
    if with_roi:
        roi_name = "roi"
        initializers.insert(
            0,
            onnx.helper.make_tensor(
                "roi", onnx.TensorProto.FLOAT, [8], [0, 0, 0, 0, 1, 1, 1, 1]
            ),
        )

    model = make_resize_model(["x", roi_name, "", "sizes"], initializers, [1, 1, 4, 4])
    x_input = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_output = ort_session.run(None, {"x": x_input})[0]

    o2p_model = ConvertModel(model)
    output = o2p_model(torch.tensor(x_input))

    np.testing.assert_allclose(
        output.detach().numpy(), exp_output, rtol=1e-5, atol=1e-5
    )


def test_convert_resize_scales_with_omitted_roi():
    scales = onnx.helper.make_tensor(
        "scales", onnx.TensorProto.FLOAT, [4], [1.0, 1.0, 2.0, 2.0]
    )
    model = make_resize_model(["x", "", "scales"], [scales], [1, 1, 4, 4])
    x_input = np.arange(4, dtype=np.float32).reshape(1, 1, 2, 2)

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_output = ort_session.run(None, {"x": x_input})[0]

    o2p_model = ConvertModel(model)
    output = o2p_model(torch.tensor(x_input))

    np.testing.assert_allclose(
        output.detach().numpy(), exp_output, rtol=1e-5, atol=1e-5
    )
