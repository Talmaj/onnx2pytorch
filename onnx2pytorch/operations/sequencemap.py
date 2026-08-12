from onnx2pytorch.operations.subgraph import SubgraphOperator


class SequenceMap(SubgraphOperator):
    def forward(
        self,
        enclosing_modules,
        enclosing_activations,
        input_sequence,
        *additional_inputs,
    ):
        """
        Parameters
        ----------
        enclosing_modules: tuple of nn.Modules
            Module(s) from enclosing scope(s), containing initializers as buffers.
        enclosing_activations: dict
            All activations from the enclosing scope.
        input_sequence: list
            Sequence whose elements the body is applied to.
        additional_inputs: list
            Tensors passed to every iteration, or sequences passed element-wise.

        Returns
        -------
        outputs: list
            One output sequence per body output.
        """
        buffer_modules = enclosing_modules + (self,)
        outputs = [[] for _ in self.output_names]

        for i, element in enumerate(input_sequence):
            activations = dict(enclosing_activations)
            activations[self.input_names[0]] = element
            for name, value in zip(self.input_names[1:], additional_inputs):
                activations[name] = value[i] if isinstance(value, list) else value

            activations = self.execute_body(buffer_modules, activations)

            for output, name in zip(outputs, self.output_names):
                output.append(activations[name])
        return outputs
