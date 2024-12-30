import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchReduceSum(nn.Module, OnnxToTorchModule):
    def __init__(self, axes, keepdims, noop_with_empty_axes):
        super().__init__()
        self.axes = axes
        self.keepdims = keepdims
        self.noop_with_empty_axes = noop_with_empty_axes

    def forward(self, x):
        if self.axes is None:
            if self.noop_with_empty_axes:
                return x
            else:
                return torch.sum(x)
        else:
            return torch.sum(x, self.axes, self.keepdims)


@converter(operation_type='ReduceSum', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    keepdims = onnx_node.attributes().get('keepdims', 1)
    noop_with_empty_axes = onnx_node.attributes().get('noop_with_empty_axes', 0)

    axes = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]) if len(onnx_node.inputs()) > 1 else None

    if axes is not None:
        axes = int(axes.to_numpy())

    return OperationConverterResult(
        torch_module=TorchReduceSum(axes, keepdims, noop_with_empty_axes),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
