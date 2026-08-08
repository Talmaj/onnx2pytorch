from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _MaxPoolNd
from onnx2pytorch.operations import (
    Attention,
    BatchNormWrapper,
    Dropout,
    GRUWrapper,
    If,
    InstanceNormWrapper,
    LinearAttention,
    Loop,
    LSTMWrapper,
    RNNWrapper,
    Split,
    TopK,
    Unique,
)


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (
    _MaxPoolNd,
    Attention,
    Dropout,
    GRUWrapper,
    If,
    LinearAttention,
    Loop,
    LSTMWrapper,
    RNNWrapper,
    Split,
    TopK,
    Unique,
)
STANDARD_LAYERS = (
    _ConvNd,
    BatchNormWrapper,
    GRUWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    RNNWrapper,
    nn.Linear,
)
