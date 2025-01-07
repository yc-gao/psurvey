from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OperationConverterResult, OnnxMapping

op_mapping = {
    1: nn.BatchNorm1d,
    2: nn.BatchNorm2d,
}


@converter(operation_type='BatchNormalization', version=15)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    epsilon = onnx_node.attributes().get('epsilon', 1e-5)
    momentum = 1 - onnx_node.attributes().get('momentum', 0.9)

    vinfo = onnx_model.get_vinfo_by_name(onnx_node.inputs()[0])
    shape = [x.dim_value if x.HasField(
        'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]
    spatial_rank = len(shape) - 2

    scale = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_torch()
    bias = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[2]).to_torch()
    input_mean = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[3]).to_torch()
    input_var = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[4]).to_torch()

    torch_cls = op_mapping[spatial_rank]
    torch_module = torch_cls(
        num_features=scale.size()[0],
        eps=epsilon,
        momentum=momentum
    )

    torch_module.weight.data = scale
    torch_module.bias.data = bias
    torch_module.running_var.data = input_var
    torch_module.running_mean.data = input_mean

    return OperationConverterResult(
        torch_module=torch_module,
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
