import torch
from torch import nn


from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import converter
from .utils import OnnxToTorchModule, OperationConverterResult, OnnxMapping


# class TorchScatterNdFunc(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, data, indices, updates) -> torch.Tensor:
#         output = data.clone()
#         ind_dim = indices.dim()
#         # last dimension is a partial-index into data
#         output_indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
#         # update.shape = indices.shape[0:ind_dim-1] ++ data.shape[indices.shape[-1]:data.dim()-1]
#         output_updates = updates.reshape((-1, *updates.shape[ind_dim - 1:]))
#         output[output_indices] = output_updates
#         return output
#
#     @staticmethod
#     def symbolic(g: torch.Graph, data, indices, updates) -> torch.Value:
#         return g.op("ScatterND", data, indices, updates, outputs=1)


class TorchScatterNd(nn.Module, OnnxToTorchModule):
    def __init__(self):
        super().__init__()

    def forward(self, data, indices, updates):
        # return TorchScatterNdFunc.apply(data, indices, updates)

        output = data.clone()
        ind_dim = indices.dim()
        # last dimension is a partial-index into data
        output_indices = indices.reshape((-1, indices.shape[-1])).T.tolist()
        # update.shape = indices.shape[0:ind_dim-1] ++ data.shape[indices.shape[-1]:data.dim()-1]
        output_updates = updates.reshape((-1, *updates.shape[ind_dim - 1:]))
        output[output_indices] = output_updates
        return output


@converter(operation_type='ScatterND', version=16)
def _(onnx_node: OnnxNode, onnx_model: OnnxModel) -> OperationConverterResult:
    reduction = onnx_node.attributes().get('reduction', 'none')
    assert reduction == 'none', 'not implement'

    return OperationConverterResult(
        torch_module=TorchScatterNd(),
        onnx_mapping=OnnxMapping(
            inputs=onnx_node.inputs(),
            outputs=onnx_node.outputs(),
        ),
    )
