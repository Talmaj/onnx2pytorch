from collections import defaultdict
from copy import deepcopy
from functools import partial
import warnings

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.jit import TracerWarning
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Identity
from torch.nn.modules.pooling import _MaxPoolNd

from onnx2pytorch.operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    Split,
)
from onnx2pytorch.convert.debug import debug_model_conversion
from onnx2pytorch.convert.operations import convert_operations
from onnx2pytorch.utils import get_inputs_names


class InitParameters(dict):
    """Use for parameters that are hidden."""

    def __getitem__(self, item):
        return super().__getitem__(item)

    def get(self, item, default):
        if item in self:
            return self[item]
        else:
            return default


class ConvertModel(nn.Module):
    def __init__(
        self, onnx_model: onnx.ModelProto, batch_dim=0, experimental=False, debug=False
    ):
        """
        Convert onnx model to pytorch.

        Parameters
        ----------
        onnx_model: onnx.ModelProto
            Loaded onnx model.
        batch_dim: int
            Dimension of the batch.
        experimental: bool
            Experimental implementation allows batch_size > 1. However,
            batchnorm layers could potentially produce false outputs.

        Returns
        -------
        model: torch.nn.Module
            A converted pytorch model.
        """
        super().__init__()
        self.onnx_model = onnx_model
        self.batch_dim = batch_dim
        self.experimental = experimental
        self.debug = debug
        self.mapping = {}
        self.device = None
        opset_version = onnx_model.opset_import[0].version
        for op_id, op_name, op in convert_operations(
            onnx_model.graph, opset_version, batch_dim
        ):
            setattr(self, op_name, op)
            self.mapping[op_id] = op_name

        self.init_parameters = InitParameters(
            {
                tensor.name: torch.from_numpy(np.copy(numpy_helper.to_array(tensor)))
                for tensor in self.onnx_model.graph.initializer
            }
        )

        self.input_names = get_inputs_names(onnx_model.graph)

        self.needed_by = defaultdict(set)
        for node in self.onnx_model.graph.node:
            out_op_id = node.output[0]
            for in_op_id in node.input:
                self.needed_by[in_op_id].add(out_op_id)
        self.needed_by.default_factory = None

        if experimental:
            warnings.warn(
                "Using experimental implementation that allows 'batch_size > 1'."
                "Batchnorm layers could potentially produce false outputs."
            )

    def forward(self, *input_list, **input_dict):
        if len(input_list) > 0 and len(input_dict) > 0:
            raise ValueError(
                "forward-pass accepts either input_list (positional args) or "
                "input_dict (keyword args) but not both"
            )
        if len(input_list) > 0:
            inputs = input_list
        if len(input_dict) > 0:
            inputs = [input_dict[key] for key in self.input_names]

        if not self.experimental and inputs[0].shape[self.batch_dim] > 1:
            raise NotImplementedError(
                "Input with larger batch size than 1 not supported yet."
            )
        activations = dict(zip(self.input_names, inputs))
        still_needed_by = deepcopy(self.needed_by)

        for node in self.onnx_model.graph.node:
            # Identifying the layer ids and names
            out_op_id = node.output[0]
            out_op_name = self.mapping[out_op_id]
            in_op_names = [
                self.mapping.get(in_op_id, in_op_id)
                for in_op_id in node.input
                if in_op_id in activations
            ]

            # getting correct layer
            op = getattr(self, out_op_name)

            # if first layer choose input as in_activations
            # if not in_op_names and len(node.input) == 1:
            #    in_activations = input
            layer_types = (
                nn.Linear,
                _ConvNd,
                BatchNormWrapper,
                InstanceNormWrapper,
                LSTMWrapper,
            )
            composite_types = (nn.Sequential,)
            multioutput_types = (_MaxPoolNd, Split, LSTMWrapper)
            if isinstance(op, layer_types) or (
                isinstance(op, composite_types)
                and any(isinstance(x, layer_types) for x in op.modules())
            ):
                in_activations = [
                    activations[in_op_id]
                    for in_op_id in node.input
                    if in_op_id in activations
                ]
            else:
                in_activations = [
                    activations[in_op_id] if in_op_id in activations
                    # if in_op_id not in activations neither in parameters then
                    # it must be the initial input
                    else self.init_parameters.get(in_op_id, inputs[0])
                    for in_op_id in node.input
                ]

            # TODO: this is only needed because some ops are apparently sending
            # activations back to the CPU. These ops should be fixed.
            in_activations = [
                in_act.to(self.device)
                for in_act in in_activations
                if in_act is not None
            ]

            # store activations for next layer
            if isinstance(op, partial) and op.func == torch.cat:
                activations[out_op_id] = op(in_activations)
            elif isinstance(op, multioutput_types):
                for out_op_id, output in zip(node.output, op(*in_activations)):
                    activations[out_op_id] = output
            elif isinstance(op, Identity):
                # After batch norm fusion the batch norm parameters
                # were all passed to identity instead of first one only
                activations[out_op_id] = op(in_activations[0])
            else:
                activations[out_op_id] = op(*in_activations)

            # Remove activations that are no longer needed
            for in_op_id in node.input:
                if in_op_id in still_needed_by:
                    still_needed_by[in_op_id].discard(out_op_id)
                    if len(still_needed_by[in_op_id]) == 0:
                        if in_op_id in activations:
                            del activations[in_op_id]

            if self.debug:
                # compare if the activations of pytorch are the same as from onnxruntime
                debug_model_conversion(
                    self.onnx_model,
                    [activations[x] for x in self.input_names],
                    [activations[out_op_id] for out_op_id in node.output],
                    node,
                )

        # collect all outputs
        outputs = [activations[x.name] for x in self.onnx_model.graph.output]
        if len(outputs) == 1:
            outputs = outputs[0]
        return outputs

    def to(self, device):
        super(ConvertModel, self).to(device=device)
        self.device = device
        for op_id in self.init_parameters:
            if self.init_parameters[op_id].device != device:
                self.init_parameters[op_id] = self.init_parameters[op_id].to(device)
        return self

    def cuda(self, device=None):
        super(ConvertModel, self).cuda(device=device)
        if device is None:
            return self.to("cuda:0")
        else:
            return self.to("cuda:{}".format(device))
