from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OperationConverterResult, onnx_mapping_from_node

op_mapping = {
    1: nn.MaxPool1d,
    2: nn.MaxPool2d,
    3: nn.MaxPool3d,
}


@converter(operation_type='MaxPool', version=12)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:  # pylint: disable=unused-argument
    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')
    ceil_mode = bool(onnx_node.attributes().get('ceil_mode', 0))
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    pads = onnx_node.attributes().get('pads')
    strides = onnx_node.attributes().get('strides')
    dilations = onnx_node.attributes().get('dilations', 1)
    storage_order = onnx_node.attributes().get('storage_order', 0)

    assert auto_pad == "NOTSET", "not implement"
    assert pads[:len(kernel_shape)] == pads[len(
        kernel_shape):], "not implement"
    assert storage_order == 0, "not implement"

    torch_cls = op_mapping[len(kernel_shape)]
    torch_module = torch_cls(
        kernel_shape,
        strides,
        pads[len(kernel_shape):],
        dilations,
        ceil_mode=ceil_mode
    )
    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=onnx_mapping_from_node(onnx_node),
    )
