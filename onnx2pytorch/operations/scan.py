import onnx
import torch

from onnx2pytorch.operations.subgraph import SubgraphOperator


def get_per_input_value(values, index, default=0):
    """Scan's per-scan-input/output attributes default to 0 when not given."""
    if values is None or index >= len(values):
        return default
    return values[index]


class Scan(SubgraphOperator):
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
        if opset_version < 9:
            # Scan-8 prepends sequence_lens and gives every tensor a batch dimension
            raise NotImplementedError(
                "Scan at opset {} not implemented.".format(opset_version)
            )
        super().__init__(opset_version, batch_dim, body)
        self.num_scan_inputs = num_scan_inputs
        self.scan_input_axes = scan_input_axes
        self.scan_input_directions = scan_input_directions
        self.scan_output_axes = scan_output_axes
        self.scan_output_directions = scan_output_directions

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

            activations = self.execute_body(buffer_modules, activations, inputs[0])

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
