from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxMapping,  OperationConverterResult

op_mapping = {
    ('Conv', 1): nn.Conv1d,
    ('Conv', 2): nn.Conv2d,
    ('ConvTranspose', 1): nn.ConvTranspose1d,
    ('ConvTranspose', 2): nn.ConvTranspose2d,
}


@converter(operation_type='Conv', version=11)
@converter(operation_type='ConvTranspose', version=11)
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

    weight = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    bias = None
    if len(onnx_node.inputs()) >= 3:
        bias = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[2]).to_torch()

    kwargs = {
        'in_channels': weight.shape[1] * group,
        'out_channels': weight.shape[0],
        'kernel_size': kernel_shape,
        'stride': strides,
        'padding': pads[len(kernel_shape):],
        'dilation': dilations,
        'groups': group,
        'bias': bias is not None,
    }

    if onnx_node.op_type() == 'ConvTranspose':
        output_padding = onnx_node.attributes().get('output_padding', 0)
        output_shape = onnx_node.attributes().get('output_shape', None)
        assert output_shape is None
        kwargs['output_padding'] = output_padding

        kwargs['in_channels'] = weight.shape[0]
        kwargs['out_channels'] = weight.shape[1] * group

    torch_module = op_mapping[(onnx_node.op_type(), len(kernel_shape))](
        **kwargs
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
