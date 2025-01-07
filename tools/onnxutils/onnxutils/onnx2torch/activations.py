from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OperationConverterResult, onnx_mapping_from_node


@converter(operation_type='Relu', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=nn.ReLU(),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )


@converter(operation_type='Sigmoid', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=nn.Sigmoid(),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
