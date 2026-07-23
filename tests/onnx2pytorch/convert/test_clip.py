import numpy as np
import onnx
import onnxruntime as ort
import torch

from onnx2pytorch.convert import ConvertModel


def test_convert_clip_missing_optional_max():
    x = onnx.helper.make_tensor_value_info(
        "x", onnx.TensorProto.FLOAT, [1]
    )
    y = onnx.helper.make_tensor_value_info(
        "y", onnx.TensorProto.FLOAT, [1]
    )

    min_tensor = onnx.helper.make_tensor(
        "min",
        onnx.TensorProto.FLOAT,
        [],
        [0.0],
    )

    # max is an optional input and is intentionally omitted
    clip_node = onnx.helper.make_node(
        "Clip",
        inputs=["x", "min", ""],
        outputs=["y"],
    )

    graph = onnx.helper.make_graph(
        [clip_node],
        "clip_missing_max",
        [x],
        [y],
        [min_tensor],
    )

    model = onnx.helper.make_model_gen_version(
        graph,
        producer_name="clip-test",
        opset_imports=[onnx.helper.make_opsetid("", 13)],
    )

    onnx.checker.check_model(model)

    x_input = np.array([-1.0], dtype=np.float32)

    # Reference output from ONNX Runtime
    ort_session = ort.InferenceSession(model.SerializeToString())
    ort_output = ort_session.run(
        None,
        {"x": x_input},
    )[0]

    # Converted PyTorch model
    o2p_model = ConvertModel(model)
    o2p_output = o2p_model(torch.tensor(x_input))

    np.testing.assert_allclose(
        o2p_output.detach().numpy(),
        ort_output,
        rtol=1e-5,
        atol=1e-5,
    )

if __name__ == "__main__":
    test_convert_clip_missing_optional_max()
