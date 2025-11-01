from functools import partial
from importlib import import_module

import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn.modules.linear import Identity

from onnx2pytorch.utils import (
    get_inputs_names,
    get_outputs_names,
)


class If(nn.Module):
    """
    If conditional operator.

    Executes then_branch if condition is true, else_branch otherwise.
    """

    def __init__(
        self,
        opset_version,
        batch_dim,
        then_branch: onnx.GraphProto,
        else_branch: onnx.GraphProto,
    ):
        super().__init__()
        self.ops = import_module("onnx2pytorch.convert.operations")
        self.c = import_module("onnx2pytorch.constants")

        self.then_branch = then_branch
        self.else_branch = else_branch
        self.batch_dim = batch_dim

        # Get input/output names for both branches
        self.then_input_names = get_inputs_names(then_branch)
        self.then_output_names = get_outputs_names(then_branch)
        self.else_input_names = get_inputs_names(else_branch)
        self.else_output_names = get_outputs_names(else_branch)

        # Create mappings for then_branch
        self.then_mapping = {}
        for op_id, op_name, op in self.ops.convert_operations(
            then_branch, opset_version, batch_dim
        ):
            then_op_name = f"then_{op_name}"
            setattr(self, then_op_name, op)
            self.then_mapping[op_id] = then_op_name

        # Create mappings for else_branch
        self.else_mapping = {}
        for op_id, op_name, op in self.ops.convert_operations(
            else_branch, opset_version, batch_dim
        ):
            else_op_name = f"else_{op_name}"
            setattr(self, else_op_name, op)
            self.else_mapping[op_id] = else_op_name

        # Store initializers for then_branch as buffers
        for tensor in self.then_branch.initializer:
            buffer_name = self.ops.get_buffer_name(f"then_{tensor.name}")
            self.register_buffer(
                buffer_name,
                torch.tensor(numpy_helper.to_array(tensor)),
            )

        # Store initializers for else_branch as buffers
        for tensor in self.else_branch.initializer:
            buffer_name = self.ops.get_buffer_name(f"else_{tensor.name}")
            self.register_buffer(
                buffer_name,
                torch.tensor(numpy_helper.to_array(tensor)),
            )

    def _execute_branch(
        self,
        branch,
        mapping,
        input_names,
        output_names,
        enclosing_modules,
        enclosing_activations,
        branch_prefix,
    ):
        """
        Execute a single branch (then or else).

        Parameters
        ----------
        branch : onnx.GraphProto
            The graph to execute.
        mapping : dict
            Mapping from output IDs to operation names.
        input_names : list
            Input names for this branch.
        output_names : list
            Output names for this branch.
        enclosing_modules : tuple
            Modules from enclosing scope.
        enclosing_activations : dict
            Activations from enclosing scope.
        branch_prefix : str
            Prefix for buffer names ("then_" or "else_").

        Returns
        -------
        list
            Output values from the branch.
        """
        buffer_modules = enclosing_modules + (self,)
        activations = {}
        activations.update(enclosing_activations)

        for node in branch.node:
            # Identify layer IDs and names
            out_op_id = node.output[0]
            out_op_name = mapping[out_op_id]

            # Get the operation
            op = getattr(self, out_op_name)

            # Determine input activations based on layer type
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
                in_activations = []
                for in_op_id in node.input:
                    if in_op_id in activations:
                        in_activations.append(activations[in_op_id])
                    else:
                        # Look for initializer with branch prefix
                        prefixed_name = f"{branch_prefix}{in_op_id}"
                        param = self.ops.get_init_parameter(
                            buffer_modules, prefixed_name, None
                        )
                        if param is not None:
                            in_activations.append(param)
                        else:
                            # Try without prefix (from enclosing scope)
                            param = self.ops.get_init_parameter(
                                buffer_modules, in_op_id, None
                            )
                            if param is not None:
                                in_activations.append(param)

            in_activations = [in_act for in_act in in_activations if in_act is not None]

            # Execute the operation and store output
            if isinstance(op, If):
                # Nested If - pass enclosing context
                outputs = op(buffer_modules, activations, *in_activations)
                for out_act_name, output in zip(node.output, outputs):
                    activations[out_act_name] = output
            elif isinstance(op, partial) and op.func == torch.cat:
                output = op(in_activations)
                activations[out_op_id] = output
            elif isinstance(op, Identity):
                output = op(in_activations[0] if in_activations else None)
                activations[out_op_id] = output
            elif isinstance(op, self.c.MULTIOUTPUT_LAYERS) or (
                isinstance(op, self.c.COMPOSITE_LAYERS)
                and any(isinstance(x, self.c.MULTIOUTPUT_LAYERS) for x in op.modules())
            ):
                outputs = op(*in_activations)
                for out_act_name, output in zip(node.output, outputs):
                    activations[out_act_name] = output
            else:
                output = op(*in_activations)
                activations[out_op_id] = output

        # Return outputs in order
        return [activations[out_name] for out_name in output_names]

    def forward(self, enclosing_modules, enclosing_activations, cond):
        """
        Execute If conditional.

        Parameters
        ----------
        enclosing_modules : tuple of nn.Modules
            Module(s) from enclosing scope(s), containing initializers as buffers.
        enclosing_activations : dict
            All activations from the enclosing scope.
        cond : torch.Tensor
            Boolean condition tensor (must contain a single element).

        Returns
        -------
        list
            Output values from the executed branch.
        """
        # Extract boolean value from condition tensor
        if isinstance(cond, torch.Tensor):
            cond_value = bool(cond.item())
        else:
            cond_value = bool(cond)

        if cond_value:
            # Execute then_branch
            return self._execute_branch(
                self.then_branch,
                self.then_mapping,
                self.then_input_names,
                self.then_output_names,
                enclosing_modules,
                enclosing_activations,
                "then_",
            )
        else:
            # Execute else_branch
            return self._execute_branch(
                self.else_branch,
                self.else_mapping,
                self.else_input_names,
                self.else_output_names,
                enclosing_modules,
                enclosing_activations,
                "else_",
            )
