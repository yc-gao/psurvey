from .onnx_model import OnnxModel

from .registry import find_optimizer


def apply_optimizers(onnx_model: OnnxModel, optimizers):
    for name in optimizers:
        optimizer = find_optimizer(name)
        onnx_model = optimizer.apply(onnx_model)
    return onnx_model
