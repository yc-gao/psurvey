from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


class TorchSlice(nn.Module, OnnxToTorchModule):
    def __init__(self, starts, ends, axes, steps):
        super().__init__()
        self.starts = starts
        self.ends = ends
        self.axes = axes
        self.steps = steps

    def forward(self, x):
        for start, end, axis, step in zip(self.starts, self.ends, self.axes, self.steps):
            if axis == -1:
                x = x[..., start:end:step]
            elif axis == 0:
                x = x[start:end:step]
            elif axis == 1:
                x = x[:, start:end:step]
            elif axis == 2:
                x = x[:, :, start:end:step]
            elif axis == 3:
                x = x[:, :, :, start:end:step]
            else:
                raise NotImplementedError
        return x


@converter(operation_type='Slice', version=13)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    starts = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[1]).to_numpy().tolist()
    ends = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[2]).to_numpy().tolist()
    axes = onnx_model.get_initializer_by_name(
        onnx_node.inputs()[3]).to_numpy().tolist()

    steps = [1] * len(axes)
    if len(onnx_node.inputs()) >= 5:
        steps = onnx_model.get_initializer_by_name(
            onnx_node.inputs()[4]).to_numpy().tolist()

    return OperationConverterResult(
        torch_module=TorchSlice(starts, ends, axes, steps),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs()[:1],
            outputs=onnx_node.outputs(),
        ),
    )
