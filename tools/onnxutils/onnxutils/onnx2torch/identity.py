from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node


class TorchIdentity(nn.Module, OnnxToTorchModule):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


@converter(operation_type='Identity', version=16)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:  # pylint: disable=unused-argument
    return OperationConverterResult(
        torch_module=TorchIdentity(),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
