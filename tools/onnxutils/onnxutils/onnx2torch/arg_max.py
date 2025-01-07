import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchArgMax(nn.Module, OnnxToTorchModule):
    def __init__(self, axis, keepdims):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        return torch.argmax(x, self.axis, self.keepdims)


@converter(operation_type='ArgMax', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axis = onnx_node.attributes().get('axis', 0)
    keepdims = bool(onnx_node.attributes().get('keepdims', 1))
    select_last_index = bool(
        onnx_node.attributes().get('select_last_index', 0))

    assert select_last_index == False, 'not implement'

    return OperationConverterResult(
        torch_module=TorchArgMax(axis, keepdims),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
