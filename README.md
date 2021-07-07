# ONNX to PyTorch
![PyPI - License](https://img.shields.io/pypi/l/onnx2pytorch?color)
[![CircleCI](https://circleci.com/gh/ToriML/onnx2pytorch.svg?style=shield)](https://app.circleci.com/pipelines/github/ToriML/onnx2pytorch)
[![Downloads](https://pepy.tech/badge/onnx2pytorch)](https://pepy.tech/project/onnx2pytorch)
![PyPI](https://img.shields.io/pypi/v/onnx2pytorch)

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
- [MobileNet](https://github.com/onnx/models/tree/master/vision/classification/mobilenet)
- [ResNet](https://github.com/onnx/models/tree/master/vision/classification/resnet)
- [ShuffleNet_V2](https://github.com/onnx/models/tree/master/vision/classification/shufflenet)
- [BERT-Squad](https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad)
- [EfficientNet-Lite4](https://github.com/onnx/models/tree/master/vision/classification/efficientnet-lite4)
- [Fast Neural Style Transfer](https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style)
- [Super Resolution](https://github.com/onnx/models/tree/master/vision/super_resolution/sub_pixel_cnn_2016)
- [YOLOv4](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov4)
  (Not exactly the same, nearest neighbour interpolation in pytorch differs)
- [U-net](https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/)
  (Converted from pytorch to onnx and then back)

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
```pre-commit install```

### Testing
[Pytest](https://docs.pytest.org/en/latest/) and [tox](https://tox.readthedocs.io/en/latest/).  
```tox```
#### Test fixtures
To test the complete conversion of an onnx model download pre-trained models: 
```./download_fixtures.sh```  
Use flag `--all` to download more models.
Add any custom models to `./fixtures` folder to test their conversion.

### Debugging
Set `ConvertModel(..., debug=True)` to compare each converted
activation from pytorch with the activation from onnxruntime.  
This helps identify where in the graph the activations start to differ.
