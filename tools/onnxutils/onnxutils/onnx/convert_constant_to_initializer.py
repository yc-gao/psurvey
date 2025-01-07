from .onnx_model import OnnxModel

from .registry import optimizer


@optimizer('convert-constant-to-initializer')
class OnnxSimplifier:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = onnx_model.output_names()
        with onnx_model.session() as sess:
            for node in onnx_model.nodes():
                if node.op_type() != 'Constant':
                    continue
                if node.outputs()[0] in output_names:
                    continue
                val = node.attributes().get('value', None)
                if val is None:
                    continue

                tensor = val.proto()
                tensor.name = node.outputs()[0]
                sess.add_initializer(tensor)
                sess.remove_node(node)

        return onnx_model
