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


def get_per_input_value(values, index, default=0):
    """Scan's per-scan-input/output attributes default to 0 when not given."""
    if values is None or index >= len(values):
        return default
    return values[index]


class Scan(nn.Module):
    def __init__(
        self,
        opset_version,
        batch_dim,
        body: onnx.GraphProto,
        num_scan_inputs,
        scan_input_axes=None,
        scan_input_directions=None,
        scan_output_axes=None,
        scan_output_directions=None,
    ):
        super().__init__()
        self.ops = import_module("onnx2pytorch.convert.operations")
        self.c = import_module("onnx2pytorch.constants")

        self.body = body
        self.batch_dim = batch_dim
        self.num_scan_inputs = num_scan_inputs
        self.scan_input_axes = scan_input_axes
        self.scan_input_directions = scan_input_directions
        self.scan_output_axes = scan_output_axes
        self.scan_output_directions = scan_output_directions

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

    def _execute_body(self, buffer_modules, activations, inputs):
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
                                buffer_modules, in_op_id, inputs[0]
                            )
                        )
                    )
                    for in_op_id in node.input
                ]

            in_activations = [in_act for in_act in in_activations if in_act is not None]
            in_activations = resolve_omitted_inputs(in_activations)

            if isinstance(op, (If, Loop, Scan)):
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

    def forward(self, enclosing_modules, enclosing_activations, *inputs):
        """
        Parameters
        ----------
        enclosing_modules: tuple of nn.Modules
            Module(s) from enclosing scope(s), containing initializers as buffers.
        enclosing_activations: dict
            All activations from the enclosing scope.
        inputs: list
            N initial values of the loop carried state, then M scan inputs.

        Returns
        -------
        outputs: list
            N final state values, then K stacked scan outputs.
        """
        num_state_vars = len(inputs) - self.num_scan_inputs
        num_scan_outputs = len(self.output_names) - num_state_vars

        state_names_in = self.input_names[:num_state_vars]
        scan_names_in = self.input_names[num_state_vars:]
        state_names_out = self.output_names[:num_state_vars]
        scan_names_out = self.output_names[num_state_vars:]

        states = list(inputs[:num_state_vars])
        scan_values = inputs[num_state_vars:]

        input_axes = [
            get_per_input_value(self.scan_input_axes, i)
            for i in range(self.num_scan_inputs)
        ]
        input_directions = [
            get_per_input_value(self.scan_input_directions, i)
            for i in range(self.num_scan_inputs)
        ]
        output_axes = [
            get_per_input_value(self.scan_output_axes, i)
            for i in range(num_scan_outputs)
        ]
        output_directions = [
            get_per_input_value(self.scan_output_directions, i)
            for i in range(num_scan_outputs)
        ]

        num_iterations = scan_values[0].shape[input_axes[0]]
        buffer_modules = enclosing_modules + (self,)
        scan_outputs = [[] for _ in scan_names_out]

        for iteration in range(num_iterations):
            activations = dict(enclosing_activations)
            activations.update(zip(state_names_in, states))
            for name, value, axis, direction in zip(
                scan_names_in, scan_values, input_axes, input_directions
            ):
                index = num_iterations - 1 - iteration if direction else iteration
                activations[name] = value.select(axis, index)

            activations = self._execute_body(buffer_modules, activations, inputs)

            states = [activations[name] for name in state_names_out]
            for i, name in enumerate(scan_names_out):
                scan_outputs[i].append(activations[name])

        outputs = states
        for values, axis, direction in zip(
            scan_outputs, output_axes, output_directions
        ):
            if direction:
                values = values[::-1]
            outputs.append(torch.stack(values, dim=axis))
        return outputs
