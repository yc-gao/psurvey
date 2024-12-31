import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchGather(nn.Module, OnnxToTorchModule):
    def __init__(self, indices, axis):
        super().__init__()
        self.indices = indices
        self.axis = axis

    def forward(self, x):
        if self.axis == 0:
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
            raise NotImplementedError(f"gather at {self.axis} not supported")


@converter(operation_type='Gather', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axis = onnx_node.attributes().get('axis', 0)
    indices = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()

    return OperationConverterResult(
        torch_module=TorchGather(indices, axis),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
