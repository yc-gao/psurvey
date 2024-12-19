from onnx_model import OnnxModel


class EliminateIdentity:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        input_name_map = {}
        for node in onnx_model.get_nodes_by_optype('Identity'):
            input_name_map[node.output[0]] = node.input[0]
        onnx_model.remap_input_names(input_name_map)
        onnx_model.remove_unused()
        onnx_model.ReInit()
        return onnx_model
