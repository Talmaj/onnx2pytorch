import warnings
import torch

from torch.nn.modules.instancenorm import _InstanceNorm

try:
    from torch.nn.modules.batchnorm import _LazyNormBase

    class _LazyInstanceNorm(_LazyNormBase, _InstanceNorm):

        cls_to_become = _InstanceNorm


except ImportError:
    from torch.nn.modules.lazy import LazyModuleMixin
    from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

    class _LazyInstanceNorm(LazyModuleMixin, _InstanceNorm):

        weight: UninitializedParameter  # type: ignore[assignment]
        bias: UninitializedParameter  # type: ignore[assignment]

        cls_to_become = _InstanceNorm

        def __init__(
            self,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            device=None,
            dtype=None,
        ) -> None:
            factory_kwargs = {"device": device, "dtype": dtype}
            super(_LazyInstanceNorm, self).__init__(
                # affine and track_running_stats are hardcoded to False to
                # avoid creating tensors that will soon be overwritten.
                0,
                eps,
                momentum,
                False,
                False,
                **factory_kwargs,
            )
            self.affine = affine
            self.track_running_stats = track_running_stats
            if self.affine:
                self.weight = UninitializedParameter(**factory_kwargs)
                self.bias = UninitializedParameter(**factory_kwargs)
            if self.track_running_stats:
                self.running_mean = UninitializedBuffer(**factory_kwargs)
                self.running_var = UninitializedBuffer(**factory_kwargs)
                self.num_batches_tracked = torch.tensor(
                    0,
                    dtype=torch.long,
                    **{k: v for k, v in factory_kwargs.items() if k != "dtype"},
                )

        def reset_parameters(self) -> None:
            if not self.has_uninitialized_params() and self.num_features != 0:
                super().reset_parameters()

        def initialize_parameters(self, input) -> None:  # type: ignore[override]
            if self.has_uninitialized_params():
                self.num_features = input.shape[1]
                if self.affine:
                    assert isinstance(self.weight, UninitializedParameter)
                    assert isinstance(self.bias, UninitializedParameter)
                    self.weight.materialize((self.num_features,))
                    self.bias.materialize((self.num_features,))
                if self.track_running_stats:
                    self.running_mean.materialize(
                        (self.num_features,)
                    )  # type:ignore[union-attr]
                    self.running_var.materialize(
                        (self.num_features,)
                    )  # type:ignore[union-attr]
                self.reset_parameters()


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
