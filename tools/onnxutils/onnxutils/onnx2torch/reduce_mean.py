import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchReduceMean(nn.Module, OnnxToTorchModule):
    def __init__(self, axes, keepdims):
        super().__init__()
        self.axes = axes
        self.keepdims = keepdims

    def forward(self, x):
        return torch.mean(x, self.axes, self.keepdims)


@converter(operation_type='ReduceMean', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axes = onnx_node.attributes().get('axes')
    keepdims = onnx_node.attributes().get('keepdims', 1)

    return OperationConverterResult(
        torch_module=TorchReduceMean(axes, keepdims),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
