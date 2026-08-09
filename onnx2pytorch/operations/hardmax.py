import torch

from onnx2pytorch.operations.softmax import NormalizingOperator


class Hardmax(NormalizingOperator):
    def normalize(self, input, dim):
        maximal = input == torch.max(input, dim=dim, keepdim=True).values
        # In case of ties only the first maximal element is set to 1
        first = torch.cumsum(maximal.to(torch.int64), dim=dim) == 1
        return torch.logical_and(maximal, first).to(input.dtype)
