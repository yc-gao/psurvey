from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('eliminate-identity')
class EliminateIdentity:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.transaction() as t:
            for node in onnx_model.get_nodes_by_optype('Identity'):
                t.remap_input_names({node.output[0]: node.input[0]})
        return onnx_model
