from onnx_model import OnnxModel
from registry import optimizer


@optimizer('eliminate-cast')
class EliminateCast:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        input_name_map = {}
        for node in reversed(onnx_model.get_nodes_by_optype('Cast')):
            i_vinfo = onnx_model.get_vinfo_by_name(node.input[0])
            if i_vinfo is None:
                continue
            o_vinfo = onnx_model.get_vinfo_by_name(node.output[0])
            if o_vinfo is None:
                continue

            if i_vinfo.type == o_vinfo.type:
                input_name_map[node.output[0]] = node.input[0]

        onnx_model.remap_input_names(input_name_map)
        return onnx_model.remove_unused()

