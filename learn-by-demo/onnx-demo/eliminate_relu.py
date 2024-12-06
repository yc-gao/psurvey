import onnx
import numpy as np

from onnx_model import OnnxModel

from matcher import DagMatcher

pattern = DagMatcher({
    'id': 1,
    'op_type': 'DequantizeLinear',
    'inputs': [
        {
            'id': 2,
            'op_type': 'QuantizeLinear',
            'inputs': [
                {
                    'id': 3,
                    'op_type': 'Relu'
                }
            ]
        }
    ]
})


class EliminateRelu:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        new_initializers = []
        input_name_map = {}

        dags = pattern.MatchAll(onnx_model)
        for dag in dags:
            relu = pattern.FindNode(dag, 3)
            q_node = pattern.FindNode(dag, 2)
            dq_node = pattern.FindNode(dag, 1)
            scale = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(q_node.input[1]))
            zp = onnx.numpy_helper.to_array(
                onnx_model.get_initializer_by_name(q_node.input[2]))
            if (zp == -128).all():
                q_node.input[0] = relu.input[0]
                input_name_map[relu.output[0]] = dq_node.output[0]
            elif (zp == 0).all():
                zp = np.ones_like(zp) * -128
                scale = scale / 2
                new_initializers.extend(
                    [(q_node.input[1], scale), (q_node.input[2], zp)])
                q_node.input[0] = relu.input[0]
                input_name_map[relu.output[0]] = dq_node.output[0]

        onnx_model.remove_initializers(
            [onnx_model.get_initializer_by_name(x[0]) for x in new_initializers])
        onnx_model.add_initializers(
            [onnx.numpy_helper.from_array(x[1], x[0]) for x in new_initializers])

        onnx_model.remap_input_names(input_name_map)
        onnx_model.remove_unused()
        onnx_model.ReInit()
        return onnx_model
