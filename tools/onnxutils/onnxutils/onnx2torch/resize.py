from torch import nn


from onnxutils.common import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping

mode_mapping = {
    ('nearest', 1): 'nearest',
    ('nearest', 2): 'nearest',
    ('nearest', 3): 'nearest',
    ('linear', 1): 'linear',
    ('linear', 2): 'bilinear',
    ('linear', 3): 'trilinear',
    ('cubic', 2): 'bicubic',
}


class TorchResize(nn.Module, OnnxToTorchModule):
    def __init__(self, scales, mode, align_corners):
        super().__init__()
        self.scales = scales
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = nn.functional.interpolate(
            x,
            size=None,
            scale_factor=self.scales,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


@converter(operation_type='Resize', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:  # pylint: disable=unused-argument
    assert len(onnx_node.inputs()) == 3, "not implement"
    roi, scales = [
        onnx_model.get_initializer_by_name(x)
        for x in onnx_node.inputs()[1:]]
    assert roi is None, "not implement"
    scales = scales.to_numpy().tolist()
    assert scales[:2] == [1, 1], "not implement"
    scales = scales[2:]

    coordinate_transformation_mode = onnx_node.attributes().get(
        'coordinate_transformation_mode', 'half_pixel')
    cubic_coeff_a = onnx_node.attributes().get('cubic_coeff_a', -0.75)
    exclude_outside = onnx_node.attributes().get('exclude_outside', 0)
    extrapolation_value = onnx_node.attributes().get('extrapolation_value', 0)
    mode = onnx_node.attributes().get('mode', 'nearest')
    nearest_mode = onnx_node.attributes().get('nearest_mode', 'round_prefer_floor')

    assert coordinate_transformation_mode == 'asymmetric', 'not implement'
    assert cubic_coeff_a == -0.75, 'not implement'
    assert exclude_outside == 0, 'not implement'
    assert extrapolation_value == 0, 'not implement'
    assert mode == 'nearest', 'not implement'
    assert nearest_mode == 'floor', 'not implement'

    torch_mode = mode_mapping[(mode, len(scales))]
    torch_align_corners = mode != 'nearest' and coordinate_transformation_mode == 'align_corners'

    return OperationConverterResult(
        torch_module=TorchResize(scales, torch_mode, torch_align_corners),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
