import io
import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from onnx2pytorch.convert import ConvertModel


def test_loop_sum():
    y_in = onnx.helper.make_tensor_value_info("y_in", onnx.TensorProto.FLOAT, [1])
    y_out = onnx.helper.make_tensor_value_info("y_out", onnx.TensorProto.FLOAT, [1])
    scan_out = onnx.helper.make_tensor_value_info(
        "scan_out", onnx.TensorProto.FLOAT, []
    )
    cond_in = onnx.helper.make_tensor_value_info("cond_in", onnx.TensorProto.BOOL, [])
    cond_out = onnx.helper.make_tensor_value_info("cond_out", onnx.TensorProto.BOOL, [])
    iter_count = onnx.helper.make_tensor_value_info(
        "iter_count", onnx.TensorProto.INT64, []
    )

    x = np.array([1, 2, 3, 4, 5]).astype(np.float32)

    x_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["x"],
        value=onnx.helper.make_tensor(
            name="const_tensor_x",
            data_type=onnx.TensorProto.FLOAT,
            dims=x.shape,
            vals=x.flatten().astype(float),
        ),
    )

    one_const_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["one"],
        value=onnx.helper.make_tensor(
            name="const_tensor_one", data_type=onnx.TensorProto.INT64, dims=(), vals=[1]
        ),
    )

    i_add_node = onnx.helper.make_node(
        "Add", inputs=["iter_count", "one"], outputs=["end"]
    )

    start_unsqueeze_node = onnx.helper.make_node(
        "Unsqueeze", inputs=["iter_count"], outputs=["slice_start"], axes=[0]
    )

    end_unsqueeze_node = onnx.helper.make_node(
        "Unsqueeze", inputs=["end"], outputs=["slice_end"], axes=[0]
    )

    slice_node = onnx.helper.make_node(
        "Slice", inputs=["x", "slice_start", "slice_end"], outputs=["slice_out"]
    )

    y_add_node = onnx.helper.make_node(
        "Add", inputs=["y_in", "slice_out"], outputs=["y_out"]
    )

    identity_node = onnx.helper.make_node(
        "Identity", inputs=["cond_in"], outputs=["cond_out"]
    )

    scan_identity_node = onnx.helper.make_node(
        "Identity", inputs=["y_out"], outputs=["scan_out"]
    )

    loop_body = onnx.helper.make_graph(
        [
            identity_node,
            x_const_node,
            one_const_node,
            i_add_node,
            start_unsqueeze_node,
            end_unsqueeze_node,
            slice_node,
            y_add_node,
            scan_identity_node,
        ],
        "loop_body",
        [iter_count, cond_in, y_in],
        [cond_out, y_out, scan_out],
    )

    node = onnx.helper.make_node(
        "Loop",
        inputs=["trip_count", "cond", "y"],
        outputs=["res_y", "res_scan"],
        body=loop_body,
    )

    trip_count = onnx.helper.make_tensor_value_info(
        "trip_count", onnx.TensorProto.INT64, []
    )
    cond = onnx.helper.make_tensor_value_info("cond", onnx.TensorProto.BOOL, [])
    y = onnx.helper.make_tensor_value_info("y", onnx.TensorProto.FLOAT, [1])
    res_y = onnx.helper.make_tensor_value_info("res_y", onnx.TensorProto.FLOAT, [1])
    res_scan = onnx.helper.make_tensor_value_info(
        "res_scan", onnx.TensorProto.FLOAT, []
    )

    graph_def = onnx.helper.make_graph(
        nodes=[node],
        name="test-model",
        inputs=[trip_count, cond, y],
        outputs=[res_y, res_scan],
    )

    model_def = onnx.helper.make_model_gen_version(
        graph_def,
        producer_name="loop-example",
        opset_imports=[onnx.helper.make_opsetid("", 11)],
    )
    onnx.checker.check_model(model_def)
    bitstream = io.BytesIO()
    onnx.save(model_def, bitstream)
    bitstream_data = bitstream.getvalue()

    trip_count_input = np.array(5).astype(np.int64)
    cond_input = np.array(1).astype(np.bool)
    y_input = np.array([-2]).astype(np.float32)
    exp_res_y = np.array([13]).astype(np.float32)
    exp_res_scan = np.array([-1, 1, 4, 8, 13]).astype(np.float32).reshape((5, 1))

    ort_session = ort.InferenceSession(bitstream_data)
    ort_inputs = {"trip_count": trip_count_input, "cond": cond_input, "y": y_input}
    ort_outputs = ort_session.run(None, ort_inputs)
    ort_res_y, ort_res_scan = ort_outputs
    np.testing.assert_allclose(ort_res_y, exp_res_y, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ort_res_scan, exp_res_scan, rtol=1e-5, atol=1e-5)

    o2p_model = ConvertModel(model_def, experimental=True)
    o2p_inputs = {
        "trip_count": torch.from_numpy(trip_count_input),
        "cond": torch.from_numpy(cond_input),
        "y": torch.from_numpy(y_input),
    }
    o2p_outputs = o2p_model(**o2p_inputs)
    o2p_res_y, o2p_res_scan = o2p_outputs
    np.testing.assert_allclose(
        o2p_res_y.detach().numpy(), exp_res_y, rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        o2p_res_scan.detach().numpy(), exp_res_scan, rtol=1e-5, atol=1e-5
    )


if __name__ == "__main__":
    test_loop_sum()
