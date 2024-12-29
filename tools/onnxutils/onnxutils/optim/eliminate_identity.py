from onnxutils.common import OnnxModel

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
                if node.output_values()[0] in output_names:
                    continue
                sess.remap_input_values(
                    {node.output_values()[0]: node.input_values()[0]})
        return onnx_model
