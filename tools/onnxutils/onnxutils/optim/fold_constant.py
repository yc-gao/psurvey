import onnx

from onnxutils.onnx import OnnxModel

from .registry import optimizer


@optimizer('fold-constant')
class FoldConstant:
    @staticmethod
    def eval_const_node(onnx_model: OnnxModel, constant_node):
        onnx_model = onnx_model.extract([], constant_node.outputs())
        import onnxruntime as ort
        sess = ort.InferenceSession(
            onnx_model.proto().SerializeToString(),
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
        output_names = set(onnx_model.output_names())
        with onnx_model.session() as sess:
            input_names = onnx_model.input_names()

            mutable_nodes = set()
            for node in onnx_model.nodes():
                if any(x in input_names for x in node.inputs()):
                    input_names.update(node.outputs())
                    mutable_nodes.add(node.name())

            nodes_to_fold = [
                x for x in onnx_model.nodes()
                if x.name() not in mutable_nodes and all(
                    output_name not in output_names
                    for output_name in x.outputs())]

            for node in nodes_to_fold:
                sess.add_initializers(
                    FoldConstant.eval_const_node(onnx_model, node))
                sess.remove_node(node)

        return onnx_model
