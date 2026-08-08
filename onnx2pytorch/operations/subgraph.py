from functools import partial
from importlib import import_module

import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn.modules.linear import Identity

from onnx2pytorch.operations.if_op import If
from onnx2pytorch.operations.loop import Loop
from onnx2pytorch.utils import (
    get_inputs_names,
    get_outputs_names,
    resolve_omitted_inputs,
    OMITTED_INPUT,
)


class SubgraphOperator(nn.Module):
    """
    Base for operators that repeatedly execute an ONNX subgraph, such as Scan
    and SequenceMap. Subclasses receive the enclosing scope in their forward
    pass and drive the body through execute_body.
    """

    def __init__(self, opset_version, batch_dim, body: onnx.GraphProto):
        super().__init__()
        self.ops = import_module("onnx2pytorch.convert.operations")
        self.c = import_module("onnx2pytorch.constants")

        self.body = body
        self.batch_dim = batch_dim

        self.input_names = get_inputs_names(body)
        self.output_names = get_outputs_names(body)

        # Creates mapping from node (identified by first output) to submodule
        self.mapping = {}
        for op_id, op_name, op in self.ops.convert_operations(
            body, opset_version, batch_dim
        ):
            setattr(self, op_name, op)
            self.mapping[op_id] = op_name

        # Store initializers as buffers
        for tensor in self.body.initializer:
            self.register_buffer(
                self.ops.get_buffer_name(tensor.name),
                torch.tensor(numpy_helper.to_array(tensor)),
            )

    def execute_body(self, buffer_modules, activations, fallback):
        """
        Run all body nodes once, adding their outputs to activations.

        Parameters
        ----------
        buffer_modules: tuple of nn.Modules
            Modules whose buffers may hold initializers.
        activations: dict
            Activations visible to the body, including its inputs.
        fallback: torch.Tensor
            Value used for inputs that resolve to neither activation nor buffer.
        """
        for node in self.body.node:
            out_op_id = node.output[0]
            op = getattr(self, self.mapping[out_op_id])

            if isinstance(op, self.c.STANDARD_LAYERS) or (
                isinstance(op, self.c.COMPOSITE_LAYERS)
                and any(isinstance(x, self.c.STANDARD_LAYERS) for x in op.modules())
            ):
                in_activations = [
                    activations[in_op_id]
                    for in_op_id in node.input
                    if in_op_id in activations
                ]
            else:
                in_activations = [
                    (
                        OMITTED_INPUT
                        if in_op_id == ""
                        else (
                            activations[in_op_id]
                            if in_op_id in activations
                            else self.ops.get_init_parameter(
                                buffer_modules, in_op_id, fallback
                            )
                        )
                    )
                    for in_op_id in node.input
                ]

            in_activations = [in_act for in_act in in_activations if in_act is not None]
            in_activations = resolve_omitted_inputs(in_activations)

            if isinstance(op, (If, Loop, SubgraphOperator)):
                outputs = op(buffer_modules, activations, *in_activations)
                for out_act_name, output in zip(node.output, outputs):
                    activations[out_act_name] = output
            elif isinstance(op, partial) and op.func == torch.cat:
                activations[out_op_id] = op(in_activations)
            elif isinstance(op, Identity):
                activations[out_op_id] = op(in_activations[0])
            elif isinstance(op, self.c.MULTIOUTPUT_LAYERS) or (
                isinstance(op, self.c.COMPOSITE_LAYERS)
                and any(isinstance(x, self.c.MULTIOUTPUT_LAYERS) for x in op.modules())
            ):
                for out_act_name, output in zip(node.output, op(*in_activations)):
                    activations[out_act_name] = output
            else:
                activations[out_op_id] = op(*in_activations)
        return activations
