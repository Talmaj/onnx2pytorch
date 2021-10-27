from collections import defaultdict
from copy import deepcopy
from functools import partial
from importlib import import_module
import warnings

import numpy as np
import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn.modules.linear import Identity

from onnx2pytorch.utils import (
    get_inputs_names,
    get_outputs_names,
)


class Loop(nn.Module):
    def __init__(
        self,
        opset_version,
        batch_dim,
        body: onnx.GraphProto,
    ):
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
            buffer_name = self.ops.get_buffer_name(tensor.name)
            self.register_buffer(
                buffer_name,
                torch.from_numpy(numpy_helper.to_array(tensor)),
            )

        # We do not track dependencies (for memory reduction) within loops.
        # This would be complicated due to loop-carried dependencies.

    def forward(self, enclosing_modules, enclosing_activations, *inputs):
        """
        Parameters
        ----------
        enclosing_modules: tuple of nn.Modules
            Module(s) from enclosing scope(s), containing initializers as buffers.
        enclosing_activations: dict
            All activations from the enclosing scope.
        inputs: list
            Inputs to Loop node (length >= 2), comprising M, cond, and v_initial.

        Returns
        -------
        v_final_and_scan_outputs: list
            Final N loop carried dependency values, then K scan_outputs.
        """

        N = len(self.input_names) - 2
        K = len(self.output_names) - (1 + N)

        M = inputs[0]
        cond = inputs[1]
        v_initial = inputs[2:]

        iteration_num_name = self.input_names[0]
        cond_in_name = self.input_names[1]
        loop_carried_in_names = self.input_names[2:]
        cond_out_name = self.output_names[0]
        loop_carried_out_names = self.output_names[1 : N + 1]
        scan_outputs_names = self.output_names[1 + N :]

        buffer_modules = enclosing_modules + (self,)

        activations = {}
        activations.update(zip(loop_carried_in_names, v_initial))
        activations.update(enclosing_activations)

        scan_outputs = defaultdict(list)
        i = torch.tensor(0)
        while i < M and cond:
            activations[iteration_num_name] = i
            activations[cond_in_name] = cond
            for node in self.body.node:
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
                        activations[in_op_id] if in_op_id in activations
                        # if in_op_id not in activations neither in parameters then
                        # it must be the initial input
                        else self.ops.get_init_parameter(
                            buffer_modules, in_op_id, inputs[0]
                        )
                        for in_op_id in node.input
                    ]

                in_activations = [
                    in_act for in_act in in_activations if in_act is not None
                ]

                # store activations for next layer
                if isinstance(op, Loop):
                    outputs = op(buffer_modules, activations, *in_activations)
                    for out_act_name, output in zip(node.output, outputs):
                        activations[out_op_id] = output
                        if out_act_name in scan_outputs_names:
                            scan_outputs[out_act_name].append(output)
                elif isinstance(op, partial) and op.func == torch.cat:
                    output = op(in_activations)
                    activations[out_op_id] = output
                    if out_op_id in scan_outputs_names:
                        scan_outputs[out_op_id].append(output)
                elif isinstance(op, Identity):
                    # After batch norm fusion the batch norm parameters
                    # were all passed to identity instead of first one only
                    output = op(in_activations[0])
                    activations[out_op_id] = op(output)
                    if out_op_id in scan_outputs_names:
                        scan_outputs[out_op_id].append(output)
                elif isinstance(op, self.c.MULTIOUTPUT_LAYERS) or (
                    isinstance(op, self.c.COMPOSITE_LAYERS)
                    and any(
                        isinstance(x, self.c.MULTIOUTPUT_LAYERS) for x in op.modules()
                    )
                ):
                    outputs = op(*in_activations)
                    for out_act_name, output in zip(node.output, outputs):
                        activations[out_act_name] = output
                        if out_act_name in scan_outputs_names:
                            scan_outputs[out_act_name].append(output)
                else:
                    output = op(*in_activations)
                    activations[out_op_id] = output
                    if out_op_id in scan_outputs_names:
                        scan_outputs[out_op_id].append(output)

            # Prepare for next iteration
            cond = activations[cond_out_name]
            i += 1
            loop_carried_outputs = [
                activations[aname] for aname in loop_carried_out_names
            ]
            activations.update(zip(loop_carried_in_names, loop_carried_outputs))

        # Set outputs to N loop carried final values, followed by K scan outputs
        outputs = [activations[lcn] for lcn in loop_carried_out_names]
        for son in scan_outputs_names:
            outputs.append(torch.cat([so.unsqueeze(dim=0) for so in scan_outputs[son]]))
        return outputs
