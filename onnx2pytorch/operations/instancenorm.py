import warnings
import torch

from torch.nn.modules.batchnorm import _LazyNormBase
from torch.nn.modules.instancenorm import _InstanceNorm


class _LazyInstanceNorm(_LazyNormBase, _InstanceNorm):

    cls_to_become = _InstanceNorm


class LazyInstanceNormUnsafe(_LazyInstanceNorm):
    """Skips dimension check."""

    def __init__(self, *args, affine=True, **kwargs):
        super().__init__(*args, affine=affine, **kwargs)

    def _check_input_dim(self, input):
        return


class InstanceNormUnsafe(_InstanceNorm):
    """Skips dimension check."""

    def __init__(self, *args, affine=True, **kwargs):
        super().__init__(*args, affine=affine, **kwargs)

    def _check_input_dim(self, input):
        return


class InstanceNormWrapper(torch.nn.Module):
    def __init__(self, torch_params, *args, affine=True, **kwargs):
        super().__init__()
        self.has_lazy = len(torch_params) == 0
        if self.has_lazy:
            self.inu = LazyInstanceNormUnsafe(*args, affine=affine, **kwargs)
        else:
            kwargs["num_features"] = torch_params[0].shape[0]
            self.inu = InstanceNormUnsafe(*args, affine=affine, **kwargs)
            keys = ["weight", "bias"]
            for key, value in zip(keys, torch_params):
                getattr(self.inu, key).data = value

    def forward(self, input, scale=None, B=None):
        if self.has_lazy:
            self.inu.initialize_parameters(input)

        if scale is not None:
            getattr(self.inu, "weight").data = scale
        if B is not None:
            getattr(self.inu, "bias").data = B

        return self.inu.forward(input)
