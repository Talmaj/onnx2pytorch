from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _MaxPoolNd
from onnx2pytorch.operations import (
    BatchNormWrapper,
    GRUWrapper,
    If,
    InstanceNormWrapper,
    Loop,
    LSTMWrapper,
    Split,
    TopK,
)


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (_MaxPoolNd, GRUWrapper, If, Loop, LSTMWrapper, Split, TopK)
STANDARD_LAYERS = (
    _ConvNd,
    BatchNormWrapper,
    GRUWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    nn.Linear,
)
