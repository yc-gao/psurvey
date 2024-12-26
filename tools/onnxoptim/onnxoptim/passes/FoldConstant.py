import onnx

from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('fold-constant')
class FoldConstant:
    @staticmethod
    def eval_const_node(onnx_model: OnnxModel, constant_node):
        onnx_model = onnx_model.extract([], constant_node.output)
        import onnxruntime as ort
        sess = ort.InferenceSession(
            onnx_model.model().SerializeToString(),
            providers=[
                x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
                if x in ort.get_available_providers()
            ])
        outputs = sess.run(None, {})
        return [
            onnx.numpy_helper.from_array(val, output.name)
            for (val, output) in zip(outputs, sess.get_outputs())
        ]

    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.transaction() as t:
            mutable_nodes = set()

            input_names = set(onnx_model.input_names())
            for node in onnx_model.nodes():
                if any(x in input_names for x in node.input):
                    input_names.update(node.output)
                    mutable_nodes.add(node.name)

            output_names = set(onnx_model.output_names())
            nodes_to_fold = [
                x for x in onnx_model.nodes()
                if x.name not in mutable_nodes and all(
                    output_name not in output_names
                    for output_name in x.output)]

            for node in nodes_to_fold:
                t.add_initializers(
                    FoldConstant.eval_const_node(onnx_model, node))
                t.remove_node(node)

        return onnx_model
