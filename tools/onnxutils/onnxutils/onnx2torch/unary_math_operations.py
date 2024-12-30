import onnx
import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node

func_mapping = {
    'Abs': torch.abs,
    'Sqrt': torch.sqrt,
    'Atan': torch.atan,
}


class TorchUnaryOp(nn.Module, OnnxToTorchModule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


@converter(operation_type='Abs', version=13)
@converter(operation_type='Sqrt', version=13)
@converter(operation_type='Atan', version=7)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=TorchUnaryOp(func_mapping[onnx_node.op_type()]),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
