import torch
import numpy as np

from onnx2pytorch.utils import get_activation_value


def debug_model_conversion(onnx_model, inputs, pred_act, node, rtol=1e-3, atol=1e-4):
    """Compare if the activations of pytorch are the same as from onnxruntime."""
    if not isinstance(inputs, list):
        raise TypeError("inputs should be in a list.")

    if not all(isinstance(x, np.ndarray) for x in inputs):
        inputs = [x.detach().cpu().numpy() for x in inputs]

    exp_act = get_activation_value(onnx_model, inputs, list(node.output))
    if isinstance(pred_act, list):
        assert len(exp_act) == len(pred_act)
        for a, b in zip(exp_act, pred_act):
            exp = torch.from_numpy(a).cpu()
            pred = b.cpu()
            assert torch.equal(torch.tensor(exp.shape), torch.tensor(pred.shape))
            assert torch.allclose(exp, pred, rtol=rtol, atol=atol)
    else:
        exp = torch.from_numpy(exp_act[0]).cpu()
        pred = pred_act.cpu()
        assert torch.equal(torch.tensor(exp.shape), torch.tensor(pred.shape))
        assert torch.allclose(exp, pred, rtol=rtol, atol=atol)
