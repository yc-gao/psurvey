import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node


class TorchConcat(nn.Module, OnnxToTorchModule):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, *args):
        return torch.cat(args, self.axis)


@converter(operation_type='Concat', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:  # pylint: disable=unused-argument
    axis = onnx_node.attributes().get('axis')
    return OperationConverterResult(
        torch_module=TorchConcat(axis),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
