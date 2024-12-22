from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('convert-constant-to-initializer')
class ConvertConstantToInitializer:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        initializer_added = []
        nodes_to_remove = []
        for node in onnx_model.get_nodes_by_optype('Constant'):
            assert len(node.attribute) == 1
            attr = node.attribute[0]
            if attr.HasField('t'):
                nodes_to_remove.append(node)
                attr.t.name = node.output[0]
                initializer_added.append(attr.t)

        onnx_model.add_initializers(initializer_added)
        onnx_model.remove_nodes(nodes_to_remove)
        return OnnxModel(onnx_model.model())
