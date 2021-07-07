from collections import defaultdict
from functools import partial

import numpy as np
import onnx
import torch
from torch import nn
from torch.nn import functional as F
from onnx import numpy_helper
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.linear import Identity
from torch.nn.modules.pooling import _MaxPoolNd


from onnx2pytorch.convert.attribute import extract_attributes
from onnx2pytorch.convert.layer import (
    convert_layer,
    convert_linear_layer,
    convert_batch_norm_layer,
    convert_instance_norm_layer,
    convert_lstm_layer,
)
from onnx2pytorch.operations import *
from onnx2pytorch.operations.base import OperatorWrapper
from onnx2pytorch.operations import Resize, Upsample
from onnx2pytorch.utils import (
    get_inputs_names,
    get_outputs_names,
    value_wrapper,
)


class Loop(nn.Module):
    def __init__(
        self,
        opset_version,
        batch_dim,
        body: onnx.GraphProto,
    ):
        super().__init__()
        self.body = body
        self.batch_dim = batch_dim

        self.input_names = get_inputs_names(body)
        self.output_names = get_outputs_names(body)

        # Creates mapping from node (identified by first output) to submodule
        self.mapping = {}
        for op_id, op_name, op in convert_operations(body, opset_version, batch_dim):
            setattr(self, op_name, op)
            self.mapping[op_id] = op_name

        # Store initializers as buffers
        for tensor in self.body.initializer:
            buffer_name = get_buffer_name(tensor.name)
            self.register_buffer(
                buffer_name,
                torch.from_numpy(np.copy(numpy_helper.to_array(tensor))),
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
                if isinstance(op, LAYER_TYPES) or (
                    isinstance(op, COMPOSITE_TYPES)
                    and any(isinstance(x, LAYER_TYPES) for x in op.modules())
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
                        else get_init_parameter(buffer_modules, in_op_id, inputs[0])
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
                elif isinstance(op, MULTIOUTPUT_TYPES) or (
                    isinstance(op, COMPOSITE_TYPES)
                    and any(isinstance(x, MULTIOUTPUT_TYPES) for x in op.modules())
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


COMPOSITE_TYPES = (nn.Sequential,)
LAYER_TYPES = (
    nn.Linear,
    _ConvNd,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
)
MULTIOUTPUT_TYPES = (_MaxPoolNd, Loop, LSTMWrapper, Split, TopK)


def get_buffer_name(param_name):
    """
    Convert name of initializer to valid name for nn.Module attribute.
    """
    return "_initializer_{}".format(param_name.replace(".", "_"))


def get_init_parameter(modules, item, default):
    """
    Look in modules for the item, and if not found return default.

    Parameters
    ----------
    modules: list of nn.Modules
        Modules whose attributes to search.
    item: str
        Name of initializer in ONNX model.
    default: torch.Tensor
        Tensor to return if item not found.
    """
    item_name = get_buffer_name(item)
    for mod in modules:
        if hasattr(mod, item_name):
            return getattr(mod, item_name)
    return default


def _deserialize_to_torch(onnx_param):
    return torch.from_numpy(np.copy(numpy_helper.to_array(onnx_param)))


def convert_operations(onnx_graph, opset_version, batch_dim=0, enable_pruning=True):
    """
    Convert onnx model operations. Yields onnx's operator_id, operator_name and
    converted pytorch operator.

    Parameters
    ----------
    onnx_graph: onnx.GraphProto
        Loaded onnx model's GraphProto.
    opset_version: int
        ONNX model's opset version.
    batch_dim: int
        Usually 0 for computer vision models and 1 for NLP models.
    enable_pruning: bool
        Track kept/pruned indices between different calls to forward pass.

    Returns
    -------
    iterator: (op_id, op_name, op)
    """
    weights = {tensor.name: tensor for tensor in onnx_graph.initializer}

    for i, node in enumerate(onnx_graph.node):
        # extract only useful inputs
        params = [weights[par_name] for par_name in node.input if par_name in weights]

        if node.op_type == "Add":
            op = Add(feature_dim=batch_dim + 1)  # 0 for CV models and 1 for NLP
        elif node.op_type == "And":
            op = OperatorWrapper(torch.logical_and)
        elif node.op_type == "AveragePool":
            op = convert_layer(node, "AvgPool")
        elif node.op_type == "BatchNormalization":
            op = convert_batch_norm_layer(node, params=params)
        elif node.op_type == "Cast":
            op = Cast(**extract_attributes(node))
        elif node.op_type == "Ceil":
            op = OperatorWrapper(torch.ceil)
        elif node.op_type == "Clip":
            op = OperatorWrapper(torch.clamp)
        elif node.op_type == "Concat":
            op = partial(torch.cat, **extract_attributes(node))
        elif node.op_type == "Constant":
            op = Constant(**extract_attributes(node))
        elif node.op_type == "ConstantOfShape":
            op = ConstantOfShape(**extract_attributes(node))
        elif node.op_type == "Conv":
            op = convert_layer(node, "Conv", params)
        elif node.op_type == "ConvTranspose":
            op = convert_layer(node, "ConvTranspose", params)
        elif node.op_type == "Div":
            op = Div()
        elif node.op_type == "Elu":
            op = nn.ELU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Equal":
            op = OperatorWrapper(torch.eq)
        elif node.op_type == "Erf":
            op = OperatorWrapper(torch.erf)
        elif node.op_type == "Exp":
            op = OperatorWrapper(torch.exp)
        elif node.op_type == "Expand":
            op = Expand()
        elif node.op_type == "Flatten":
            op = Flatten(**extract_attributes(node))
            op.feature_dim = batch_dim + 1  # Necessary for transformers
        elif node.op_type == "Floor":
            op = OperatorWrapper(torch.floor)
        elif node.op_type == "Gather":
            op = Gather(**extract_attributes(node))
        elif node.op_type == "GatherND":
            op = GatherND(**extract_attributes(node))
        elif node.op_type == "Gemm":
            op = convert_linear_layer(node, params)
        elif node.op_type == "GlobalAveragePool":
            op = GlobalAveragePool()
        elif node.op_type == "Greater":
            op = OperatorWrapper(torch.greater)
        elif node.op_type == "Identity":
            op = nn.Identity()
        elif node.op_type == "InstanceNormalization":
            op = convert_instance_norm_layer(node, params=params)
        elif node.op_type == "LeakyRelu":
            op = nn.LeakyReLU(**extract_attributes(node), inplace=True)
        elif node.op_type == "Less":
            op = OperatorWrapper(torch.less)
        elif node.op_type == "Log":
            op = OperatorWrapper(torch.log)
        elif node.op_type == "Loop":
            op = Loop(
                opset_version=opset_version,
                batch_dim=batch_dim,
                **extract_attributes(node),
            )
        elif node.op_type == "LSTM":
            op = convert_lstm_layer(node, weights)
        elif node.op_type == "MatMul":
            if params:
                weight = _deserialize_to_torch(params[0])
                op = nn.Linear(weight.shape[0], weight.shape[1], bias=False)
                op.weight.data = weight.t()

                # check if next node Add to add bias
                next_node = onnx_graph.node[i + 1]
                next_params = [
                    weights[par_name]
                    for par_name in next_node.input
                    if par_name in weights
                ]
                if next_params and next_node.op_type == "Add":
                    bias = _deserialize_to_torch(next_params[0])
                    op.bias = nn.Parameter(bias)
                    node.output.pop()
                    node.output.extend(next_node.output)
                    onnx_graph.node.pop(i + 1)  # remove next node
            else:
                op = MatMul()
        elif node.op_type == "Max":
            op = OperatorWrapper(torch.max)
        elif node.op_type == "MaxPool":
            op = convert_layer(node, "MaxPool")
        elif node.op_type == "Min":
            op = OperatorWrapper(torch.min)
        elif node.op_type == "Mul":
            op = OperatorWrapper(torch.mul)
        elif node.op_type == "NonMaxSuppression":
            op = NonMaxSuppression(**extract_attributes(node))
        elif node.op_type == "Not":
            op = OperatorWrapper(torch.logical_not)
        elif node.op_type == "OneHot":
            op = OneHot(**extract_attributes(node))
        elif node.op_type == "Or":
            op = OperatorWrapper(torch.logical_or)
        elif node.op_type == "Pad":
            op = Pad(**extract_attributes(node))
        elif node.op_type == "Pow":
            op = OperatorWrapper(torch.pow)
        elif node.op_type == "PRelu":
            op = PRelu()
        elif node.op_type == "Range":
            op = Range()
        elif node.op_type == "Reciprocal":
            op = OperatorWrapper(torch.reciprocal)
        elif node.op_type == "ReduceMax":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.max, **kwargs)
        elif node.op_type == "ReduceMean":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.mean, **kwargs)
        elif node.op_type == "ReduceMin":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.min, **kwargs)
        elif node.op_type == "ReduceProd":
            kwargs = dict(keepdim=True)
            kwargs.update(extract_attributes(node))
            op = partial(torch.prod, **kwargs)
        elif node.op_type == "ReduceSum":
            op = ReduceSum(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Relu":
            op = nn.ReLU(inplace=True)
        elif node.op_type == "Reshape":
            shape = list(
                filter(lambda x: x.name == node.input[1], onnx_graph.initializer)
            )
            shape = np.copy(numpy_helper.to_array(shape[0])) if shape else None
            op = Reshape(enable_pruning, shape)
        elif node.op_type == "Resize":
            op = Resize(**extract_attributes(node))
        elif node.op_type == "Scatter":
            op = Scatter(**extract_attributes(node))
        elif node.op_type == "ScatterElements":
            op = ScatterElements(**extract_attributes(node))
        elif node.op_type == "ScatterND":
            op = ScatterND()
        elif node.op_type == "Shape":
            op = Shape()
        elif node.op_type == "Sigmoid":
            op = nn.Sigmoid()
        elif node.op_type == "Slice":
            op = Slice(**extract_attributes(node))
        elif node.op_type == "Softmax":
            kwargs = dict(dim=-1)
            kwargs.update(extract_attributes(node))
            op = nn.Softmax(**kwargs)
        elif node.op_type == "Softplus":
            op = nn.Softplus(beta=1)
        elif node.op_type == "Softsign":
            op = nn.Softsign()
        elif node.op_type == "Split":
            kwargs = extract_attributes(node)
            # if the split_size_or_sections is not in node attributes,
            # the number_of_splits becomes the number of node outputs
            if "split_size_or_sections" not in kwargs:
                kwargs["number_of_splits"] = len(node.output)
            op = Split(enable_pruning, **kwargs)
        elif node.op_type == "Sqrt":
            op = OperatorWrapper(torch.sqrt)
        elif node.op_type == "Squeeze":
            op = Squeeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Sub":
            op = OperatorWrapper(torch.sub)
        elif node.op_type == "Tanh":
            op = OperatorWrapper(torch.tanh)
        elif node.op_type == "ThresholdedRelu":
            op = ThresholdedRelu(**extract_attributes(node))
        elif node.op_type == "Tile":
            op = Tile()
        elif node.op_type == "TopK":
            op = TopK()
        elif node.op_type == "Transpose":
            op = Transpose(**extract_attributes(node))
        elif node.op_type == "Unsqueeze":
            op = Unsqueeze(opset_version=opset_version, **extract_attributes(node))
        elif node.op_type == "Upsample":
            op = Upsample(**extract_attributes(node))
        elif node.op_type == "Where":
            op = Where()
        else:
            op = getattr(torch, node.op_type.lower(), None)
            if op is None:
                raise NotImplementedError(
                    "Conversion not implemented for op_type={}.".format(node.op_type)
                )
            else:
                print(
                    "Automatic inference of operator: {}".format(node.op_type.lower())
                )

        op_name = "{}_{}".format(node.op_type, node.output[0])
        op_id = node.output[0]
        yield op_id, op_name, op
