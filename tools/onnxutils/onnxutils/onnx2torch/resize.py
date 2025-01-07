import warnings

from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

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
    def __init__(self, scales, sizes, mode):
        super().__init__()
        self.scales = scales
        self.sizes = sizes
        self.mode = mode

    def forward(self, x):
        x = nn.functional.interpolate(
            x,
            size=self.sizes,
            scale_factor=self.scales,
            mode=self.mode,
        )
        return x


class TorchResize1(nn.Module, OnnxToTorchModule):
    def __init__(self,  sizes):
        super().__init__()
        self.sizes = sizes

    def forward(self, x):
        x = nn.functional.upsample_bilinear(x, self.sizes)
        return x


# TODO: yinchao check later
@converter(operation_type='Resize', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:  # pylint: disable=unused-argument
    coordinate_transformation_mode = onnx_node.attributes().get(
        'coordinate_transformation_mode', 'half_pixel')
    cubic_coeff_a = onnx_node.attributes().get('cubic_coeff_a', -0.75)
    exclude_outside = onnx_node.attributes().get('exclude_outside', 0)
    extrapolation_value = onnx_node.attributes().get('extrapolation_value', 0)
    mode = onnx_node.attributes().get('mode', 'nearest')
    nearest_mode = onnx_node.attributes().get('nearest_mode', 'round_prefer_floor')

    assert onnx_node.inputs()[1] == '', "not implement"

    scales = onnx_model.get_initializer_by_name(onnx_node.inputs()[2])
    if scales is not None:
        scales = scales.to_numpy().tolist()
        assert scales[:2] == [1, 1], "not implement"
        scales = tuple(scales[2:])
        torch_mode = mode_mapping[(mode, len(scales))]

    sizes = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[3]) if len(onnx_node.inputs()) >= 4 else None
    if sizes is not None:
        sizes = sizes.to_numpy().tolist()
        vinfo = onnx_model.get_vinfo_by_name(onnx_node.inputs()[0])
        shape = [x.dim_value if x.HasField(
            'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]
        assert sizes[:2] == shape[:2], "not implement"
        sizes = tuple(sizes[2:])
        torch_mode = mode_mapping[(mode, len(sizes))]

    if coordinate_transformation_mode == 'asymmetric':
        if cubic_coeff_a != -0.75:
            warnings.warn('results might differ significantly!')
        if exclude_outside != 0:
            warnings.warn('results might differ significantly!')
        if extrapolation_value != 0:
            warnings.warn('results might differ significantly!')
        if mode == 'nearest' and nearest_mode != 'floor':
            warnings.warn('results might differ significantly!')
        return OperationConverterResult(
            torch_module=nn.Upsample(
                size=sizes,
                scale_factor=scales,
                mode=torch_mode,
            ),
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
            ),
        )
    elif coordinate_transformation_mode == 'half_pixel':
        if cubic_coeff_a != -0.75:
            warnings.warn('results might differ significantly!')
        if exclude_outside != 0:
            warnings.warn('results might differ significantly!')
        if extrapolation_value != 0:
            warnings.warn('results might differ significantly!')
        if mode == 'nearest' and nearest_mode != 'floor':
            warnings.warn('results might differ significantly!')
        return OperationConverterResult(
            torch_module=TorchResize(scales, sizes, torch_mode),
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
            ),
        )
    elif coordinate_transformation_mode == 'align_corners':
        if cubic_coeff_a != -0.75:
            warnings.warn('results might differ significantly!')
        if exclude_outside != 0:
            warnings.warn('results might differ significantly!')
        if extrapolation_value != 0:
            warnings.warn('results might differ significantly!')
        if mode != 'linear':
            warnings.warn('results might differ significantly!')
        assert sizes and len(sizes) == 2, "not implement"
        return OperationConverterResult(
            torch_module=TorchResize1(sizes),
            onnx_mapping=OnnxMapping(
                inputs=onnx_node.inputs()[:1],
                outputs=onnx_node.outputs(),
            ),
        )
    else:
        raise NotImplementedError()
