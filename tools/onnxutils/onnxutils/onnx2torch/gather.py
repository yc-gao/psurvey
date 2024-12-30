from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, onnx_mapping_from_node


class TorchGather(nn.Module, OnnxToTorchModule):
    def __init__(self, indices, axis):
        super().__init__()
        self.indices = indices
        self.axis = axis

    def forward(self, x):
        x = torch.gather(x, self.axis, self.indices)
        return x


@converter(operation_type='Gather', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    axis = onnx_node.attributes().get('axis', 0)
    indices = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()

    return OperationConverterResult(
        torch_module=TorchGather(indices, axis),
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
