from onnx_model import OnnxModel


class NormlizeModel:
    @staticmethod
    def constant2initializer(onnx_model: OnnxModel) -> OnnxModel:
        initializer_added = []
        node_removed = []

        for node in onnx_model.nodes():
            if node.op_type == 'Constant':
                assert len(node.attribute) == 1
                attr = node.attribute[0]
                if attr.HasField('t'):
                    attr.t.name = node.output[0]
                    initializer_added.append(attr.t)
                    node_removed.append(node)

        onnx_model.remove_nodes(node_removed)
        onnx_model.add_initializers(initializer_added)
        return onnx_model

    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        onnx_model = NormlizeModel.constant2initializer(onnx_model)
        return onnx_model
