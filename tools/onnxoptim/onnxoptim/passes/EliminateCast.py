from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('eliminate-cast')
class EliminateCast:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.transaction() as t:
            output_names = set(onnx_model.output_names())
            for node in onnx_model.get_nodes_by_optype('Cast'):
                if node.output[0] in output_names:
                    continue
                i_vinfo = onnx_model.get_vinfo_by_name(node.input[0])
                if i_vinfo is None:
                    continue
                o_vinfo = onnx_model.get_vinfo_by_name(node.output[0])
                if o_vinfo is None:
                    continue
                if i_vinfo.type != o_vinfo.type:
                    continue

                t.remap_input_names({node.output[0]: node.input[0]})
                t.remove_node(node)

        return onnx_model
