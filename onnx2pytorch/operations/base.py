from abc import ABC

from torch import nn


class Operator(nn.Module, ABC):
    @staticmethod
    def get_axis(input_shape, input_feature_axis):
        """
        Parameters
        ----------
        input_shape: torch.Size
        input_feature_axis: int

        Returns
        -------
        axis: tuple
            Axis to aggregate over.
        """
        if input_feature_axis < 0:
            input_feature_axis += len(input_shape)
            # select and sum all axes except the feature one
        axis = set(range(len(input_shape))) - {input_feature_axis}
        return tuple(axis)


class OperatorWrapper(Operator, ABC):
    def __init__(self, op):
        """
        This class enables any function to become a subclass of nn.Module
        The module reports itself under the op's name.

        Parameters
        ----------
        op: function or builtin_function_or_method
            Any torch function. It is used in-place of forward method.
        """
        self.forward = op
        self.op_name = op.__name__
        super().__init__()

    def _get_name(self):
        # Per instance, renaming the class would rename every other wrapper too
        return self.op_name
