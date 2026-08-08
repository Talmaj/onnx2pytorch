from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _MaxPoolNd
from onnx2pytorch.operations import (
    Attention,
    BatchNormWrapper,
    CausalConvWithState,
    Dropout,
    GRUWrapper,
    If,
    InstanceNormWrapper,
    LinearAttention,
    Loop,
    LSTMWrapper,
    RNNWrapper,
    Split,
    SubgraphOperator,
    TopK,
    Unique,
)


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (
    _MaxPoolNd,
    Attention,
    CausalConvWithState,
    Dropout,
    GRUWrapper,
    If,
    LinearAttention,
    Loop,
    LSTMWrapper,
    RNNWrapper,
    Split,
    SubgraphOperator,
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
