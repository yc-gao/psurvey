from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchMatMul(nn.Module, OnnxToTorchModule):
    def __init__(self):
        super().__init__()

    def forward(self, x0, x1):
        return x0 @ x1


@converter(operation_type='MatMul', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    return OperationConverterResult(
        torch_module=TorchMatMul(),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
