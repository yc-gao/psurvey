import torch
from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OperationConverterResult, OnnxMapping


@converter(operation_type='BatchNormalization', version=15)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    epsilon = onnx_node.attributes().get('epsilon', 1e-5)
    momentum = onnx_node.attributes().get('momentum', 0.9)

    scale = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    B = onnx_model.get_initializer_by_name(onnx_node.inputs()[2]).to_torch()
    input_mean = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[3]).to_torch()
    input_var = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[4]).to_torch()

    vinfo = onnx_model.get_vinfo_by_name(onnx_node.outputs()[0])
    shape = [x.dim_value if x.HasField(
        'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]

    if len(shape) == 4:
        torch_module = nn.BatchNorm2d(shape[1], eps=epsilon, momentum=momentum)
        torch_module.weight.data = scale
        torch_module.bias.data = B
        torch_module.running_var.data = input_var
        torch_module.running_mean.data = input_mean

        return OperationConverterResult(
            torch_module=torch_module,
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
            ),
        )
    raise NotImplementedError
