import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchPermute(nn.Module, OnnxToTorchModule):
    def __init__(self, perm):
        super().__init__()
        self.perm = perm

    def forward(self, x):
        return torch.permute(x, self.perm)


@converter(operation_type='Transpose', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    perm = onnx_node.attributes().get('perm')

    return OperationConverterResult(
        torch_module=TorchPermute(perm),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
