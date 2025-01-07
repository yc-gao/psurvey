import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchGather(nn.Module, OnnxToTorchModule):
    def __init__(self, axis, indices):
        super().__init__()
        self.axis = axis
        self.indices = indices

    def forward(self, x):
        if self.axis == -1:
            return x[..., self.indices]
        elif self.axis == 0:
            return x[self.indices]
        elif self.axis == 1:
            return x[:, self.indices]
        elif self.axis == 2:
            return x[:, :, self.indices]
        elif self.axis == 3:
            return x[:, :, :, self.indices]
        elif self.axis == 4:
            return x[:, :, :, :, self.indices]
        else:
            raise NotImplementedError


@converter(operation_type='Gather', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axis = onnx_node.attributes().get('axis', 0)
    indices = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()

    return OperationConverterResult(
        torch_module=TorchGather(axis, indices),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
