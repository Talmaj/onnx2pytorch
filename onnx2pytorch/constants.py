from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _MaxPoolNd
from onnx2pytorch.operations import (
    Attention,
    BatchNormWrapper,
    CausalConvWithState,
    ConvTranspose,
    Dropout,
    DynamicQuantizeLinear,
    GRUWrapper,
    If,
    InstanceNormWrapper,
    LayerNorm,
    LinearAttention,
    Loop,
    LSTMWrapper,
    MaxPool,
    RNNWrapper,
    SoftmaxCrossEntropyLoss,
    Split,
    StringSplit,
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
    DynamicQuantizeLinear,
    GRUWrapper,
    If,
    LayerNorm,
    LinearAttention,
    Loop,
    LSTMWrapper,
    MaxPool,
    RNNWrapper,
    SoftmaxCrossEntropyLoss,
    Split,
    StringSplit,
    SubgraphOperator,
    TopK,
    Unique,
)
STANDARD_LAYERS = (
    _ConvNd,
    BatchNormWrapper,
    ConvTranspose,
    GRUWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    RNNWrapper,
    nn.Linear,
)
