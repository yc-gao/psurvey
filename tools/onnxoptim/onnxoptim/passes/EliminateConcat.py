from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('eliminate-concat')
class EliminateConcat:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.transaction() as t:
            output_names = set(onnx_model.output_names())
            for node in onnx_model.get_nodes_by_optype('Concat'):
                if len(node.input) != 1:
                    continue
                if node.output[0] in output_names:
                    continue
                t.remap_input_names({node.output[0]: node.input[0]})
                t.remove_node(node)

        return onnx_model
