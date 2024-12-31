import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchCumSum(nn.Module, OnnxToTorchModule):
    def __init__(self, axis, exclusive, reverse):
        super().__init__()
        self.axis = axis
        self.exclusive = exclusive
        self.reverse = reverse

    def forward(self, x):
        return torch.cumsum(x, self.axis)


@converter(operation_type='CumSum', version=14)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    exclusive = onnx_node.attributes().get('exclusive', 0)
    reverse = onnx_node.attributes().get('reverse', 0)

    assert exclusive == 0, 'not implement'
    assert reverse == 0, 'not implement'

    axis = int(onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy())

    return OperationConverterResult(
        torch_module=TorchCumSum(axis, exclusive, reverse),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
