from functools import partial
from importlib import import_module

import onnx
import torch
from onnx import numpy_helper
from torch import nn
from torch.nn.modules.linear import Identity

from onnx2pytorch.operations.loop import Loop
from onnx2pytorch.utils import (
    get_inputs_names,
    get_outputs_names,
    resolve_omitted_inputs,
    OMITTED_INPUT,
)


class SubgraphOperator(nn.Module):
    """
    Base for operators that execute one or more ONNX subgraphs, such as If,
    Scan and SequenceMap. Subclasses receive the enclosing scope in their
    forward pass and drive their subgraphs through execute_graph.
    """

    def __init__(
        self,
        opset_version,
        batch_dim,
        body: onnx.GraphProto = None,
        enable_pruning=False,
    ):
        super().__init__()
        self.ops = import_module("onnx2pytorch.convert.operations")
        self.c = import_module("onnx2pytorch.constants")

        self.opset_version = opset_version
        self.batch_dim = batch_dim
        self.enable_pruning = enable_pruning

        if body is not None:
            self.body = body
            self.input_names = get_inputs_names(body)
            self.output_names = get_outputs_names(body)
            self.mapping = self.add_subgraph(body)

    @property
    def subgraph_mappings(self):
        """Pairs of subgraph and the mapping from node id to submodule name."""
        return ((self.body, self.mapping),)

    def add_subgraph(self, graph, prefix=""):
        """Convert a subgraph's nodes to submodules and its initializers to buffers."""
        mapping = {}
        for op_id, op_name, op in self.ops.convert_operations(
            graph, self.opset_version, self.batch_dim, self.enable_pruning
        ):
            submodule_name = prefix + op_name
            setattr(self, submodule_name, op)
            mapping[op_id] = submodule_name

        for tensor in graph.initializer:
            self.register_buffer(
                self.ops.get_buffer_name(prefix + tensor.name),
                torch.tensor(numpy_helper.to_array(tensor)),
            )
        return mapping

    def execute_body(self, buffer_modules, activations):
        return self.execute_graph(self.body, self.mapping, buffer_modules, activations)

    def execute_graph(self, graph, mapping, buffer_modules, activations, prefix=""):
        """
        Run all nodes of a subgraph once, adding their outputs to activations.

        Parameters
        ----------
        graph: onnx.GraphProto
            Subgraph to execute.
        mapping: dict
            Mapping from node id to the name of the corresponding submodule.
        buffer_modules: tuple of nn.Modules
            Modules whose buffers may hold initializers.
        activations: dict
            Activations visible to the subgraph, including its inputs.
        prefix: str
            Prefix under which this subgraph's initializers were registered.
        """

        def resolve(in_op_id):
            if in_op_id == "":
                return OMITTED_INPUT
            if in_op_id in activations:
                return activations[in_op_id]
            if prefix:
                param = self.ops.get_init_parameter(
                    buffer_modules, prefix + in_op_id, None
                )
                if param is not None:
                    return param
            return self.ops.get_init_parameter(buffer_modules, in_op_id)

        for node in graph.node:
            out_op_id = node.output[0]
            op = getattr(self, mapping[out_op_id])

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
                in_activations = [resolve(in_op_id) for in_op_id in node.input]

            in_activations = [in_act for in_act in in_activations if in_act is not None]
            in_activations = resolve_omitted_inputs(in_activations)

            if isinstance(op, (Loop, SubgraphOperator)):
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
