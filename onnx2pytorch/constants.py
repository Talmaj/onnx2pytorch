from torch import nn
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.pooling import _MaxPoolNd
from onnx2pytorch.operations import (
    BatchNormWrapper,
    InstanceNormWrapper,
    Loop,
    LSTMWrapper,
    Split,
    TopK,
)


COMPOSITE_LAYERS = (nn.Sequential,)
MULTIOUTPUT_LAYERS = (_MaxPoolNd, Loop, LSTMWrapper, Split, TopK)
STANDARD_LAYERS = (
    _ConvNd,
    BatchNormWrapper,
    InstanceNormWrapper,
    LSTMWrapper,
    nn.Linear,
)
