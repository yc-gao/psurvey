import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchWhere(nn.Module, OnnxToTorchModule):
    def __init__(self):
        super().__init__()

    def forward(self, cond, x, y):
        return torch.where(cond, x, y)


@converter(operation_type='Where', version=16)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=TorchWhere(),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
