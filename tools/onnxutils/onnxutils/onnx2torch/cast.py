import onnx
import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping

dtype_mapping = {
    int(onnx.TensorProto.FLOAT): torch.float32,
    int(onnx.TensorProto.UINT8): torch.uint8,
    int(onnx.TensorProto.INT8): torch.int8,
    int(onnx.TensorProto.INT16): torch.int16,
    int(onnx.TensorProto.INT32): torch.int32,
    int(onnx.TensorProto.INT64): torch.int64,
    int(onnx.TensorProto.BOOL): torch.bool,
    int(onnx.TensorProto.FLOAT16): torch.float16,
    int(onnx.TensorProto.DOUBLE): torch.float64,
    int(onnx.TensorProto.COMPLEX64): torch.complex64,
    int(onnx.TensorProto.COMPLEX128): torch.complex128,
}


class TorchCast(nn.Module, OnnxToTorchModule):
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype

    def forward(self, x):
        return x.to(self.dtype)


@converter(operation_type='Cast', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    dtype = dtype_mapping[onnx_node.attributes().get('to')]
    return OperationConverterResult(
        torch_module=TorchCast(dtype),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
