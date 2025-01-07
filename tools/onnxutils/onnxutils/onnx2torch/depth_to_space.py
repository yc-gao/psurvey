from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


@converter(operation_type='DepthToSpace', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    mode = onnx_node.attributes().get('mode', 'CRD')
    blocksize = onnx_node.attributes().get('blocksize')

    assert mode == 'CRD', 'not implement'
    torch_module = nn.PixelShuffle(blocksize)

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
