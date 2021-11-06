import warnings

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
    def __init__(self, *args, spatial=True, **kwargs):
        if not spatial:
            warnings.warn("Non-spatial BatchNorm not implemented.", RuntimeWarning)
        super().__init__(*args, **kwargs)

    def _check_input_dim(self, input):
        return


class BatchNormUnsafe(_BatchNorm):
    def __init__(self, *args, spatial=True, **kwargs):
        if not spatial:
            warnings.warn("Non-spatial BatchNorm not implemented.", RuntimeWarning)
        super().__init__(*args, **kwargs)

    def _check_input_dim(self, input):
        return


class BatchNormWrapper(nn.Module):
    def __init__(self, torch_params, *args, **kwargs):
        super().__init__()
        self.has_lazy = len(torch_params) == 0
        if self.has_lazy:
            self.bnu = LazyBatchNormUnsafe(*args, **kwargs)
        else:
            kwargs["num_features"] = torch_params[0].shape[0]
            self.bnu = BatchNormUnsafe(*args, **kwargs)
            keys = ["weight", "bias", "running_mean", "running_var"]
            for key, value in zip(keys, torch_params):
                getattr(self.bnu, key).data = value

    def forward(self, X, scale=None, B=None, input_mean=None, input_var=None):
        if self.has_lazy:
            self.bnu.initialize_parameters(X)

        if scale is not None:
            getattr(self.bnu, "weight").data = scale
        if B is not None:
            getattr(self.bnu, "bias").data = scale
        if input_mean is not None:
            getattr(self.bnu, "running_mean").data = input_mean
        if input_var is not None:
            getattr(self.bnu, "running_var").data = input_var

        return self.bnu.forward(X)
