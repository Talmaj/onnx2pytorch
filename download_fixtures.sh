mkdir -p fixtures
cd fixtures

if [[ ! -f mobilenetv2-1.0.onnx ]]; then
  echo Downloading mobilenetv2-1.0
  curl -o mobilenetv2-1.0.onnx https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.onnx
fi

#if [[ ! -f resnet18v1.onnx ]]; then
#  echo Downloading resnet18v1
#  curl -o resnet18v1.onnx https://s3.amazonaws.com/onnx-model-zoo/resnet/resnet18v1/resnet18v1.onnx
#fi

if [[ ! -f shufflenet_v2.onnx ]]; then
  echo Downloading shufflenet_v2
  curl -LJo shufflenet_v2.onnx https://github.com/onnx/models/blob/master/vision/classification/shufflenet/model/shufflenet-v2-10.onnx\?raw\=true
fi

#if [[ ! -f bertsquad-10.onnx ]]; then
#  echo Downloading bertsquad-10
#  curl -LJo bertsquad-10.onnx https://github.com/onnx/models/blob/master/text/machine_comprehension/bert-squad/model/bertsquad-10.onnx\?raw\=true
#fi

#if [[ ! -f yolo_v4.onnx ]]; then
#  echo Downloading yolo_v4
#  curl -LJo yolo_v4.onnx https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/model/yolov4.onnx\?raw\=true
#fi
