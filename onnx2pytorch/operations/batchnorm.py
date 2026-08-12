from torch import nn
from torch.nn.modules.batchnorm import _BatchNorm

try:
    from torch.nn.modules.batchnorm import _LazyNormBase

    class _LazyBatchNorm(_LazyNormBase, _BatchNorm):

        cls_to_become = _BatchNorm

except ImportError:
    # for torch < 1.10.0
    from torch.nn.modules.batchnorm import _LazyBatchNorm


class LazyBatchNormUnsafe(_LazyBatchNorm):
    def _check_input_dim(self, input):
        return


class BatchNormUnsafe(_BatchNorm):
    def _check_input_dim(self, input):
        return


def _check_spatial(spatial, scale):
    """Reject per-element statistics, which torch's batch norm cannot express.

    Opset 6 to 8 exporters, mxnet in particular, emit spatial=0 next to
    per-channel 1D parameters, where it is indistinguishable from spatial=1.
    Only parameters shaped like the input past the batch dimension really ask
    for one statistic per element.
    """
    if not spatial and scale is not None and scale.ndim > 1:
        raise NotImplementedError("BatchNormalization with spatial=0.")


class BatchNormWrapper(nn.Module):
    def __init__(self, torch_params, *args, spatial=True, **kwargs):
        super().__init__()
        self.spatial = bool(spatial)
        self.has_lazy = len(torch_params) == 0
        if self.has_lazy:
            self.bnu = LazyBatchNormUnsafe(*args, **kwargs)
        else:
            _check_spatial(self.spatial, torch_params[0])
            kwargs["num_features"] = torch_params[0].shape[0]
            self.bnu = BatchNormUnsafe(*args, **kwargs)
            keys = ["weight", "bias", "running_mean", "running_var"]
            for key, value in zip(keys, torch_params):
                getattr(self.bnu, key).data = value

        # Set to eval mode to use running statistics (ONNX inference behavior)
        self.bnu.eval()

    def forward(self, X, scale=None, B=None, input_mean=None, input_var=None):
        _check_spatial(self.spatial, scale)

        if self.has_lazy:
            self.bnu.initialize_parameters(X)

        if scale is not None:
            getattr(self.bnu, "weight").data = scale
        if B is not None:
            getattr(self.bnu, "bias").data = B
        if input_mean is not None:
            getattr(self.bnu, "running_mean").data = input_mean
        if input_var is not None:
            getattr(self.bnu, "running_var").data = input_var

        return self.bnu.forward(X)
