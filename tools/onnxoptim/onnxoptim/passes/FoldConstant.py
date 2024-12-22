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
        input_names = set(onnx_model.input_names())
        mutable_nodes = set()
        for node in onnx_model.nodes():
            if any(x in input_names for x in node.input):
                input_names.update(node.output)
                mutable_nodes.add(node.name)

        nodes_to_fold = [
            x for x in onnx_model.nodes()
            if x.name not in mutable_nodes and all(
                output_name not in onnx_model.output_names()
                for output_name in x.output)]

        initializers_to_add = []
        for node in nodes_to_fold:
            initializers_to_add.extend(
                FoldConstant.eval_const_node(onnx_model, node))

        onnx_model.add_initializers(initializers_to_add)
        onnx_model.remove_nodes(nodes_to_fold)
        return onnx_model.finalize()
