import onnx

from onnx_model import OnnxModel
from registry import optimizer


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
        mutable_nodes = set()

        input_names = set(onnx_model.input_names())
        for node in onnx_model.nodes():
            if any(x in input_names for x in node.input):
                input_names.update(node.output)
                mutable_nodes.add(node.name)

        constant_nodes = [
            x for x in onnx_model.nodes() if x.name not in mutable_nodes
        ]

        new_initializers = []
        for constant_node in constant_nodes:
            new_initializers.extend(
                FoldConstant.eval_const_node(onnx_model, constant_node))

        onnx_model.remove_nodes(constant_nodes)
        onnx_model.add_initializers(new_initializers)
        return OnnxModel(onnx_model.model())
