from onnx_model import OnnxModel
from registry import optimizer


@optimizer('convert-constant-to-initializer')
class ConvertConstantToInitializer:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        initializer_added = []
        constant_removed = []
        for node in onnx_model.get_nodes_by_optype('Constant'):
            assert len(node.attribute) == 1
            attr = node.attribute[0]
            if attr.HasField('t'):
                constant_removed.append(node)
                attr.t.name = node.output[0]
                initializer_added.append(attr.t)

        onnx_model.add_initializers(initializer_added)
        onnx_model.remove_nodes(constant_removed)
        return OnnxModel(onnx_model.model())
