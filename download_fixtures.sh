#!/usr/bin/env bash
mkdir -p fixtures
cd fixtures

if [[ ! -f mobilenetv2-1.0.onnx ]]; then
  echo Downloading mobilenetv2-1.0
  curl -o mobilenetv2-1.0.onnx https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx
fi

if [[ ! -f shufflenet_v2.onnx ]]; then
  echo Downloading shufflenet_v2
  curl -LJo shufflenet_v2.onnx https://github.com/onnx/models/blob/master/vision/classification/shufflenet/model/shufflenet-v2-10.onnx\?raw\=true
fi

if [[ $1 == "--all" ]]; then
  if [[ ! -f resnet18v1.onnx ]]; then
    echo Downloading resnet18v1
    curl -o resnet18v1.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx
  fi

  if [[ ! -f bertsquad-10.onnx ]]; then
    echo Downloading bertsquad-10
    curl -LJo bertsquad-10.onnx https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx\?raw\=true
  fi

  if [[ ! -f yolo_v4.onnx ]]; then
    echo Downloading yolo_v4
    curl -LJo yolo_v4.onnx https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx\?raw\=true
  fi

  if [[ ! -f super_res.onnx ]]; then
    echo Downloading super_res
    curl -LJo super_res.onnx https://github.com/onnx/models/blob/master/vision/super_resolution/sub_pixel_cnn_2016/model/super-resolution-10.onnx\?raw\=true
  fi

  if [[ ! -f fast_neural_style.onnx ]]; then
    echo Downloading fast_neural_style
    curl -LJo fast_neural_style.onnx https://github.com/onnx/models/blob/master/vision/style_transfer/fast_neural_style/model/rain-princess-9.onnx\?raw\=true
  fi

  if [[ ! -f efficientnet-lite4.onnx ]]; then
    echo Downloading efficientnet-lite4
    curl -LJo efficientnet-lite4.onnx https://github.com/onnx/models/blob/master/vision/classification/efficientnet-lite4/model/efficientnet-lite4-11.onnx\?raw\=true
  fi

  if [[ ! -f mobilenetv2-7.onnx ]]; then
    echo Downloading mobilenetv2-7
    curl -LJo mobilenetv2-7.onnx https://github.com/onnx/models/raw/master/vision/classification/mobilenet/model/mobilenetv2-7.onnx\?raw\=true
  fi
 
fi

echo "All models downloaded."
