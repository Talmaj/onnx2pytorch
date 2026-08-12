import onnx
import torch

from onnx2pytorch.operations.subgraph import SubgraphOperator
from onnx2pytorch.utils import get_inputs_names, get_outputs_names


class If(SubgraphOperator):
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
        enable_pruning=False,
    ):
        super().__init__(opset_version, batch_dim, enable_pruning=enable_pruning)

        self.then_branch = then_branch
        self.else_branch = else_branch

        self.then_input_names = get_inputs_names(then_branch)
        self.then_output_names = get_outputs_names(then_branch)
        self.else_input_names = get_inputs_names(else_branch)
        self.else_output_names = get_outputs_names(else_branch)

        self.then_mapping = self.add_subgraph(then_branch, "then_")
        self.else_mapping = self.add_subgraph(else_branch, "else_")

    @property
    def subgraph_mappings(self):
        return (
            (self.then_branch, self.then_mapping),
            (self.else_branch, self.else_mapping),
        )

    def forward(self, enclosing_modules, enclosing_activations, cond):
        """
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
        if isinstance(cond, torch.Tensor):
            taken = bool(cond.item())
        else:
            taken = bool(cond)

        if taken:
            branch, mapping = self.then_branch, self.then_mapping
            output_names, prefix = self.then_output_names, "then_"
        else:
            branch, mapping = self.else_branch, self.else_mapping
            output_names, prefix = self.else_output_names, "else_"

        activations = self.execute_graph(
            branch,
            mapping,
            enclosing_modules + (self,),
            dict(enclosing_activations),
            prefix,
        )
        return [activations[name] for name in output_names]
