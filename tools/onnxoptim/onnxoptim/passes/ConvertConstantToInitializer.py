from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('convert-constant-to-initializer')
class ConvertConstantToInitializer:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.transaction() as t:
            for node in onnx_model.get_nodes_by_optype('Constant'):
                assert len(node.attribute) == 1
                attr = node.attribute[0]
                if attr.HasField('t'):
                    attr.t.name = node.output[0]

                    t.add_initializer(attr.t)
                    t.remove_node(node)

        return onnx_model
