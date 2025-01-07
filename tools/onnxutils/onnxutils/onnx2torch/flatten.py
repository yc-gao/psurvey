from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OperationConverterResult, OnnxMapping


@converter(operation_type='Flatten', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axis = onnx_node.attributes().get('axis', 1)

    return OperationConverterResult(
        torch_module=nn.Flatten(axis),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
