import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchClip(nn.Module, OnnxToTorchModule):
    def __init__(self, min_val=None, max_val=None):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, x):
        return torch.clip(x, self.min_val, self.max_val)


@converter(operation_type='Clip', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    min_val = onnx_model.get_initializer_by_name(onnx_node.inputs()[1])
    max_val = onnx_model.get_initializer_by_name(onnx_node.inputs()[2])

    if min_val is not None:
        min_val = float(min_val.to_numpy())
    if max_val is not None:
        max_val = float(max_val.to_numpy())

    return OperationConverterResult(
        torch_module=TorchClip(min_val, max_val),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
