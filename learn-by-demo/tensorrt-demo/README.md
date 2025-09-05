# Install

```shell
apt install -y libnvinfer-dev libnvonnxparsers-dev libnvinfer-bin
python -m pip install tensorrt cuda-python
```

```shell
trtexec --stronglyTyped --onnx=resnet50.onnx --saveEngine=resnet50.trt
```
