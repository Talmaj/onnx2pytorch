import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from onnx2pytorch.convert import ConvertModel


def make_clip_model(with_min, with_max):
    x = onnx.helper.make_tensor_value_info("x", onnx.TensorProto.FLOAT, [3])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [3])

    initializers = []
    if with_min:
        initializers.append(
            onnx.helper.make_tensor("min", onnx.TensorProto.FLOAT, [], [0.0])
        )
    if with_max:
        initializers.append(
            onnx.helper.make_tensor("max", onnx.TensorProto.FLOAT, [], [1.0])
        )

    # Omitted optional inputs are marked with an empty input name
    clip_node = onnx.helper.make_node(
        "Clip",
        inputs=["x", "min" if with_min else "", "max" if with_max else ""],
        outputs=["y"],
    )

    graph = onnx.helper.make_graph([clip_node], "clip", [x], [y], initializers)
    model = onnx.helper.make_model_gen_version(
        graph,
        producer_name="clip-test",
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )
    onnx.checker.check_model(model)
    return model


@pytest.mark.parametrize(
    "with_min, with_max",
    [(True, False), (False, True), (True, True), (False, False)],
)
def test_convert_clip_omitted_optional_inputs(with_min, with_max):
    model = make_clip_model(with_min, with_max)
    x_input = np.array([-5.0, 0.5, 5.0], dtype=np.float32)

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_output = ort_session.run(None, {"x": x_input})[0]

    o2p_model = ConvertModel(model)
    output = o2p_model(torch.tensor(x_input))

    np.testing.assert_allclose(
        output.detach().numpy(), exp_output, rtol=1e-5, atol=1e-5
    )
