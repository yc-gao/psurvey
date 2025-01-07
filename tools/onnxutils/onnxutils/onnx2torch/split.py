import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchSplit(nn.Module, OnnxToTorchModule):
    def __init__(self, axis, split):
        super().__init__()
        self.axis = axis
        self.split = split

    def forward(self, x):
        return torch.split(x, self.split, self.axis)


@converter(operation_type='Split', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axis = onnx_node.attributes().get('axis', 0)
    split = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()

    return OperationConverterResult(
        torch_module=TorchSplit(axis, split),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
