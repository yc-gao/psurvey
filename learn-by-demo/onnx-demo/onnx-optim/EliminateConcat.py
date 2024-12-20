from onnx_model import OnnxModel
from registry import optimizer


@optimizer('eliminate-concat')
class EliminateConcat:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        input_name_map = {}
        for node in reversed(onnx_model.get_nodes_by_optype('Concat')):
            if len(node.input) == 1:
                input_name_map[node.output[0]] = node.input[0]

        onnx_model.remap_input_names(input_name_map)
        return onnx_model.remove_unused()
