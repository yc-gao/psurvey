import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node


class TorchBinaryOp(nn.Module, OnnxToTorchModule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x0, x1):
        return self.f(x0, x1)


@converter(operation_type='Add', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=TorchBinaryOp(torch.add),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
