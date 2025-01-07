from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

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
    weight = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    bias = None
    if len(onnx_node.inputs()) >= 3:
        bias = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[2]).to_torch()

    spatial_rank = len(weight.shape) - 2

    auto_pad = onnx_node.attributes().get('auto_pad', 'NOTSET')
    group = onnx_node.attributes().get('group', 1)
    kernel_shape = onnx_node.attributes().get('kernel_shape', weight.shape[2:])
    dilations = onnx_node.attributes().get(
        'dilations', [1] * spatial_rank)
    pads = onnx_node.attributes().get('pads', [0, 0] * spatial_rank)
    strides = onnx_node.attributes().get('strides', [1] * spatial_rank)

    assert auto_pad == 'NOTSET', "not implement"
    assert pads[:spatial_rank] == pads[spatial_rank:], "not implement"

    kwargs = {
        'in_channels': weight.shape[1] * group,
        'out_channels': weight.shape[0],
        'kernel_size': kernel_shape,
        'stride': strides,
        'padding': pads[:spatial_rank],
        'dilation': dilations,
        'groups': group,
        'bias': bias is not None,
    }

    if onnx_node.op_type() == 'ConvTranspose':
        output_padding = onnx_node.attributes().get(
            'output_padding', [0] * spatial_rank)
        kwargs['output_padding'] = output_padding

        output_shape = onnx_node.attributes().get('output_shape', None)
        assert output_shape is None

        kwargs['in_channels'] = weight.shape[0]
        kwargs['out_channels'] = weight.shape[1] * group

    torch_cls = op_mapping[(onnx_node.op_type(), spatial_rank)]
    torch_module = torch_cls(
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
