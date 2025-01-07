from .onnx_model import OnnxModel

from .registry import optimizer


@optimizer('eliminate-identity')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = onnx_model.output_names()
        with onnx_model.session() as sess:
            for node in onnx_model.nodes():
                if node.op_type() != 'Identity':
                    continue
                if node.outputs()[0] in output_names:
                    continue
                sess.remap_node_inputs(
                    {node.outputs()[0]: node.inputs()[0]})
        return onnx_model
