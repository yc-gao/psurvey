import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchPad(nn.Module, OnnxToTorchModule):
    def __init__(self, mode, pads, constant_value):
        super().__init__()
        self.mode = mode
        self.pads = pads
        self.constant_value = constant_value

    def forward(self, x):
        return nn.functional.pad(x, self.pads, self.mode, self.constant_value)


@converter(operation_type='Pad', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    mode = onnx_node.attributes().get('mode', 'constant')

    pads = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().reshape(2, -1).T[::-1, :].reshape(-1).tolist()
    constant_value = 0
    if onnx_node.inputs()[2]:
        constant_value = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[2]).to_numpy().item()

    return OperationConverterResult(
        torch_module=TorchPad(mode, pads, constant_value),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
