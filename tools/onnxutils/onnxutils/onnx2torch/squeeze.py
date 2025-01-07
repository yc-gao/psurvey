import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchSqueeze(nn.Module, OnnxToTorchModule):
    def __init__(self, axis):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        return torch.squeeze(x, self.axis)


@converter(operation_type='Squeeze', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axis = int(onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy())

    return OperationConverterResult(
        torch_module=TorchSqueeze(axis),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
