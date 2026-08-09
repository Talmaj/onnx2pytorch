import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from onnx2pytorch.convert import ConvertModel
from tests.onnx2pytorch.differential import (
    assert_matches_oracle,
    make_single_node_model,
    run_converted,
    run_oracle,
)


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


SCALES = np.array([1.0, 1.0, 2.0, 1.5], dtype=np.float32)


@pytest.mark.parametrize("mode", ["nearest", "linear", "cubic"])
@pytest.mark.parametrize("opset_version", [11, 13, 18, 19])
def test_resize_modes(opset_version, mode):
    """ONNX mode names have to be mapped onto torch's rank-specific names."""
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Resize",
        {"x": x},
        opset_version,
        input_names=["x", "", "scales"],
        initializers={"scales": SCALES},
        mode=mode,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("spatial_dims", [1, 2, 3])
def test_resize_linear_by_rank(spatial_dims):
    np.random.seed(0)
    x = np.random.randn(*([1, 2] + [4] * spatial_dims)).astype(np.float32)
    scales = np.array([1.0, 1.0] + [2.0] * spatial_dims, dtype=np.float32)
    model = make_single_node_model(
        "Resize",
        {"x": x},
        13,
        input_names=["x", "", "scales"],
        initializers={"scales": scales},
        mode="linear",
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_resize_opset_10_takes_scales_as_second_input(mode):
    """Resize-10 has the signature (X, scales), not (X, roi, scales, sizes)."""
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Resize",
        {"x": x},
        10,
        input_names=["x", "scales"],
        initializers={"scales": SCALES},
        mode=mode,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_upsample_scales_input(mode):
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Upsample",
        {"x": x},
        9,
        input_names=["x", "scales"],
        initializers={"scales": SCALES},
        mode=mode,
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_upsample_scales_attribute(mode):
    """Upsample-7 passes the scales as an attribute."""
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Upsample", {"x": x}, 7, mode=mode, scales=[1.0, 1.0, 2.0, 1.5]
    )
    assert_matches_oracle(model, {"x": x})


@pytest.mark.parametrize("mode", ["nearest", "linear"])
def test_upsample_height_and_width_scale(mode):
    """Upsample-1 names the two spatial scales individually."""
    np.random.seed(0)
    x = np.random.randn(1, 2, 4, 5).astype(np.float32)
    model = make_single_node_model(
        "Upsample", {"x": x}, 1, mode=mode, height_scale=2.0, width_scale=1.5
    )
    expected = make_single_node_model(
        "Upsample", {"x": x}, 7, mode=mode, scales=[1.0, 1.0, 2.0, 1.5]
    )
    np.testing.assert_allclose(
        run_converted(model, {"x": x})[0],
        run_oracle(expected, {"x": x})[0],
        rtol=1e-5,
        atol=1e-6,
    )
