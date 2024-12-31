from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxMapping,  OperationConverterResult


@converter(operation_type='Conv', version=11)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')
    dilations = onnx_node.attributes().get('dilations')
    group = onnx_node.attributes().get('group', 1)
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    pads = onnx_node.attributes().get('pads')
    strides = onnx_node.attributes().get('strides')

    assert auto_pad == 'NOTSET', "not implement"
    assert pads[:len(kernel_shape)] == pads[len(
        kernel_shape):], "not implement"

    if len(kernel_shape) == 2:
        weight = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[1]).to_torch()
        bias = None
        if len(onnx_node.inputs()) > 2:
            bias = onnx_model.get_initializer_by_name(
                onnx_node.inputs()[2]).to_torch()

        torch_module = nn.Conv2d(
            weight.shape[1] * group,
            weight.shape[0],
            kernel_shape,
            strides,
            pads[len(kernel_shape):],
            dilations,
            group,
            bias is not None
        )
        torch_module.weight.data = weight
        if bias is not None:
            torch_module.bias.data = bias

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
            )
        )
    elif len(kernel_shape) == 1:
        weight = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[1]).to_torch()
        bias = None
        if len(onnx_node.inputs()) > 2:
            bias = onnx_model.get_initializer_by_name(
                onnx_node.inputs()[2]).to_torch()

        torch_module = nn.Conv1d(
            weight.shape[1] * group,
            weight.shape[0],
            kernel_shape,
            strides,
            pads[len(kernel_shape):],
            dilations,
            group,
            bias is not None
        )
        torch_module.weight.data = weight
        if bias is not None:
            torch_module.bias.data = bias

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
            )
        )


@converter(operation_type='ConvTranspose', version=11)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')
    dilations = onnx_node.attributes().get('dilations')
    group = onnx_node.attributes().get('group', 1)
    kernel_shape = onnx_node.attributes().get('kernel_shape')
    pads = onnx_node.attributes().get('pads')
    strides = onnx_node.attributes().get('strides')

    output_padding = onnx_node.attributes().get('output_padding', None)
    output_shape = onnx_node.attributes().get('output_shape', None)
    assert output_padding is None
    assert output_shape is None

    assert auto_pad == 'NOTSET', "not implement"
    assert pads[:len(kernel_shape)] == pads[len(
        kernel_shape):], "not implement"

    if len(kernel_shape) == 2:
        weight = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[1]).to_torch()
        bias = None
        if len(onnx_node.inputs()) > 2:
            bias = onnx_model.get_initializer_by_name(
                onnx_node.inputs()[2]).to_torch()

        torch_module = nn.ConvTranspose2d(
            weight.shape[0],
            weight.shape[1] * group,
            kernel_shape,
            strides,
            pads[len(kernel_shape):],
            groups=group,
            bias=bias is not None,
            dilation=dilations,
        )
        torch_module.weight.data = weight
        if bias is not None:
            torch_module.bias.data = bias

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
            )
        )
