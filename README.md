# ONNX to Pytorch
[![CircleCI](https://circleci.com/gh/Talmaj/onnx2pytorch.svg?style=shield&circle-token=68dcb1fcf9acf2d5de3c1160d6c1f81e25b88f35)](https://app.circleci.com/pipelines/github/Talmaj/onnx2pytorch)

A library to transform ONNX model to Pytorch. This library enables use of Pytorch 
backend and all of its great features for manipulation of neural networks.

## Installation
```pip install onnx2pytorch```

## Usage
```
import onnx
from onnx2pytorch import ConvertModel

onnx_model = onnx.load(path_to_onnx_model)
pytorch_model = ConvertModel(onnx_model)
```

Currently supported and tested models from [onnx_zoo](https://github.com/onnx/models):
- MobileNet
- ResNet
- ShuffleNet
- Bert

## Development
### Dependency installation
```pip install -r requirements.txt```

From onnxruntime>=1.5.0 you need to add the 
following to your .bashrc or .zshrc if you are running OSx:
```export KMP_DUPLICATE_LIB_OK=True```

### Code formatting
The Uncompromising Code Formatter: [Black](https://github.com/psf/black)  
```black {source_file_or_directory}```  

Install it into pre-commit hook to always commit nicely formatted code:  
```pre-commmit install```

### Testing
[Pytest](https://docs.pytest.org/en/latest/) and [tox](https://tox.readthedocs.io/en/latest/).  
```tox```