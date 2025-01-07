import onnx
import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node

func_mapping = {
    'Abs': torch.abs,
    'Neg': torch.neg,
    'Sqrt': torch.sqrt,
    'Exp': torch.exp,
    'Floor': torch.floor,
    'Tanh': torch.tanh,
    'Atan': torch.atan,
    'Cos': torch.cos,
    'Sin': torch.sin,
}


class TorchUnaryOp(nn.Module, OnnxToTorchModule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x):
        return self.f(x)


@converter(operation_type='Abs', version=13)
@converter(operation_type='Neg', version=13)
@converter(operation_type='Sqrt', version=13)
@converter(operation_type='Exp', version=13)
@converter(operation_type='Floor', version=13)
@converter(operation_type='Tanh', version=13)
@converter(operation_type='Atan', version=7)
@converter(operation_type='Cos', version=7)
@converter(operation_type='Sin', version=7)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=TorchUnaryOp(func_mapping[onnx_node.op_type()]),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
