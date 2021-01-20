# ONNX to PyTorch
[![CircleCI](https://circleci.com/gh/ToriML/onnx2pytorch.svg?style=shield)](https://app.circleci.com/pipelines/github/ToriML/onnx2pytorch)

A library to transform ONNX model to PyTorch. This library enables use of PyTorch 
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

## Limitations
Known current version limitations are:
- `batch_size > 1` could deliver unexpected results due to ambiguity of onnx's BatchNorm layer.   
That is why in this case for now we raise an assertion error.  
Set `experimental=True` in `ConvertModel` to be able to use `batch_size > 1`.
- Fine tuning and training of converted models was not tested yet, only inference.

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