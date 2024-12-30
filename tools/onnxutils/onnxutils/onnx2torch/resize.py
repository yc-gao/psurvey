import warnings

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
    def __init__(self, scales, sizes, mode, align_corners):
        super().__init__()
        self.scales = scales
        self.sizes = sizes
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = nn.functional.interpolate(
            x,
            size=self.sizes,
            scale_factor=self.scales,
            mode=self.mode,
            align_corners=self.align_corners,
        )
        return x


@converter(operation_type='Resize', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:  # pylint: disable=unused-argument
    print(onnx_node.name())
    coordinate_transformation_mode = onnx_node.attributes().get(
        'coordinate_transformation_mode', 'half_pixel')
    cubic_coeff_a = onnx_node.attributes().get('cubic_coeff_a', -0.75)
    exclude_outside = onnx_node.attributes().get('exclude_outside', 0)
    extrapolation_value = onnx_node.attributes().get('extrapolation_value', 0)
    mode = onnx_node.attributes().get('mode', 'nearest')
    nearest_mode = onnx_node.attributes().get('nearest_mode', 'round_prefer_floor')

    if mode == 'nearest':
        if nearest_mode != 'floor':
            warnings.warn(
                'Pytorch\'s nearest neighbor interpolate uses the "floor" nearest_mode. '
                'For others modes, the results might differ significantly!'
            )

        if coordinate_transformation_mode != 'asymmetric':
            warnings.warn(
                'Pytorch\'s nearest neighbor interpolation uses "asymmetric" coordinate_transformation_mode. '
                'For others modes, the results might differ significantly!'
            )
    else:
        if coordinate_transformation_mode not in ['pytorch_half_pixel', 'half_pixel']:
            warnings.warn(
                'For linear and cubic interpolation in "asymmetric" and "align_corners" coordinate_transformation_mode'
                'results might differ significantly!'
            )

    if cubic_coeff_a != -0.75:
        warnings.warn(
            'With a cubic coefficient value other than 0.75, the results might differ significantly!')

    if exclude_outside != 0:
        warnings.warn(
            'With a exclude outside value other than 0, the results might differ significantly!')

    if extrapolation_value != 0.0:
        warnings.warn(
            'With a extrapolation value other than 0.0, the results might differ significantly!')

    roi, scales, sizes = ([
        onnx_model.get_initializer_by_name(x)
        for x in onnx_node.inputs()[1:]] + [None])[:3]
    assert roi is None, "not implement"
    if scales is not None:
        scales = scales.to_numpy().tolist()
        assert scales[:2] == [1, 1], "not implement"
        scales = scales[2:]
        torch_mode = mode_mapping[(mode, len(scales))]
    if sizes is not None:
        sizes = sizes.to_numpy().tolist()
        vinfo = onnx_model.get_vinfo_by_name(onnx_node.inputs()[0])
        shape = [x.dim_value if x.HasField(
            'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]
        assert sizes[:2] == shape[:2], "not implement"
        sizes = sizes[2:]
        torch_mode = mode_mapping[(mode, len(sizes))]

    torch_align_corners = coordinate_transformation_mode == 'align_corners'
    return OperationConverterResult(
        torch_module=TorchResize(
            scales, sizes, torch_mode, torch_align_corners),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
