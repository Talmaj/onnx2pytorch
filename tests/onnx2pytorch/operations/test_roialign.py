import numpy as np
import onnxruntime as ort
import pytest
import torch
from onnx import helper, TensorProto

from onnx2pytorch.convert import ConvertModel


def check_roi_align(x, rois, batch_indices, opset_version=16, **attrs):
    node = helper.make_node(
        "RoiAlign", inputs=["x", "rois", "batch_indices"], outputs=["y"], **attrs
    )
    graph = helper.make_graph(
        [node],
        "roialign_test",
        [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, list(x.shape)),
            helper.make_tensor_value_info("rois", TensorProto.FLOAT, list(rois.shape)),
            helper.make_tensor_value_info(
                "batch_indices", TensorProto.INT64, list(batch_indices.shape)
            ),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", opset_version)]
    )

    ort_session = ort.InferenceSession(model.SerializeToString())
    exp_y = ort_session.run(
        None, {"x": x, "rois": rois, "batch_indices": batch_indices}
    )[0]

    o2p_model = ConvertModel(model)
    with torch.no_grad():
        y = o2p_model(
            torch.from_numpy(x),
            torch.from_numpy(rois),
            torch.from_numpy(batch_indices),
        )

    np.testing.assert_allclose(y.numpy(), exp_y, rtol=1e-4, atol=1e-5)


def make_inputs():
    np.random.seed(0)
    x = np.random.randn(2, 3, 10, 10).astype(np.float32)
    rois = np.array(
        [[0.0, 0.0, 4.0, 4.0], [1.5, 2.0, 7.5, 8.0], [2.0, 1.0, 9.0, 6.0]],
        dtype=np.float32,
    )
    batch_indices = np.array([0, 1, 0], dtype=np.int64)
    return x, rois, batch_indices


@pytest.mark.parametrize(
    "coordinate_transformation_mode", ["half_pixel", "output_half_pixel"]
)
@pytest.mark.parametrize("sampling_ratio", [0, 2])
def test_roi_align(coordinate_transformation_mode, sampling_ratio):
    x, rois, batch_indices = make_inputs()
    check_roi_align(
        x,
        rois,
        batch_indices,
        output_height=3,
        output_width=3,
        sampling_ratio=sampling_ratio,
        coordinate_transformation_mode=coordinate_transformation_mode,
    )


def test_roi_align_spatial_scale():
    x, rois, batch_indices = make_inputs()
    check_roi_align(
        x,
        rois,
        batch_indices,
        output_height=2,
        output_width=2,
        spatial_scale=0.5,
        sampling_ratio=2,
    )


def test_roi_align_opset10_defaults():
    x, rois, batch_indices = make_inputs()
    check_roi_align(
        x,
        rois,
        batch_indices,
        opset_version=10,
        output_height=2,
        output_width=2,
        sampling_ratio=2,
    )
