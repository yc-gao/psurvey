import onnx
import numpy as np

from onnx_model import OnnxModel

from matcher import DagMatcher

pattern = DagMatcher({
    'id': 1,
    'op_type': 'BatchNormalization',
    'inputs': [
        {
            'id': 2,
            'op_type': 'Gemm',
        }
    ]
})


class MergeGemmBN:
    @staticmethod
    def FindAttr(node, attr_name):
        for attr in node.attribute:
            if attr_name == attr.name:
                return attr
        return None

    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        new_initializers = []
        input_name_map = {}

        dags = pattern.MatchAll(onnx_model)
        for dag in dags:
            gemm = pattern.FindNode(dag, 2)
            if len(onnx_model.get_nodes_by_input_name(gemm.output[0])) > 1:
                continue
            bn = pattern.FindNode(dag, 1)
            eps = MergeGemmBN.FindAttr(bn, 'epsilon').f
            scale = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(bn.input[1]))
            bias = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(bn.input[2]))
            input_mean = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(bn.input[3]))
            input_var = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(bn.input[4]))

            factor = scale / np.sqrt(input_var + eps)
            offset = bias - input_mean * scale / np.sqrt(input_var + eps)

            transB = MergeGemmBN.FindAttr(gemm, 'transB')
            B = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(gemm.input[1]))
            C = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(gemm.input[2]))
            C = C + offset
            B = B * (np.expand_dims(factor, -1)
                     if transB and transB.i else factor)

            new_initializers.extend([(gemm.input[1], B), (gemm.input[2], C)])
            input_name_map[bn.output[0]] = gemm.output[0]

        onnx_model.remove_initializers(
            [onnx_model.get_initializer_by_name(x[0]) for x in new_initializers])
        onnx_model.add_initializers(
            [onnx.numpy_helper.from_array(x[1], x[0]) for x in new_initializers])

        onnx_model.remap_input_names(input_name_map)
        onnx_model.remove_unused()
        onnx_model.ReInit()
        return onnx_model
