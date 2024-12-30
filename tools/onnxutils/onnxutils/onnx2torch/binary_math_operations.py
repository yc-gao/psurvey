import onnx
import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node

func_mapping = {
    'Add': torch.add,
    'Greater': torch.gt,
    'Div': torch.div,
    'Div_int': lambda a, b: torch.div(a, b, rounding_mode='trunc')
}


class TorchBinaryOp(nn.Module, OnnxToTorchModule):
    def __init__(self, f):
        super().__init__()
        self.f = f

    def forward(self, x0, x1):
        return self.f(x0, x1)


@converter(operation_type='Greater', version=13)
@converter(operation_type='Add', version=14)
@converter(operation_type='Div', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    op_type = onnx_node.op_type()
    if op_type == 'Div':
        inputs = [
            onnx_model.get_vinfo_by_name(x).type.tensor_type.elem_type
            for x in onnx_node.inputs()
        ]
        integer_types = (
            onnx.TensorProto.DataType.UINT8,
            onnx.TensorProto.DataType.INT8,
            onnx.TensorProto.DataType.UINT16,
            onnx.TensorProto.DataType.INT16,
            onnx.TensorProto.DataType.UINT32,
            onnx.TensorProto.DataType.INT32,
            onnx.TensorProto.DataType.UINT64,
            onnx.TensorProto.DataType.INT64,
            onnx.TensorProto.DataType.UINT4,
            onnx.TensorProto.DataType.INT4,
        )
        if all(x in integer_types for x in inputs):
            op_type = 'Div_int'

    return OperationConverterResult(
        torch_module=TorchBinaryOp(func_mapping[op_type]),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
