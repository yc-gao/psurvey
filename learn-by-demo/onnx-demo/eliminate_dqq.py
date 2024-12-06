import numpy as np
import onnx

from onnx_model import OnnxModel
from matcher import DagMatcher

dqq_matcher = DagMatcher({
    'id': 1,
    'op_type': 'DequantizeLinear',
    'inputs': [
        {
            'id': 2,
            'op_type': 'QuantizeLinear',
        }
    ]
})


class EliminateDqQ:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        input_name_map = {}

        dags = dqq_matcher.MatchAll(onnx_model)
        for dag in dags:
            q_node = dqq_matcher.FindNode(dag, 2)
            if onnx_model.get_initializer_by_name(q_node.input[0]):
                dq_node = dqq_matcher.FindNode(dag, 1)
                input_name_map[dq_node.output[0]] = q_node.input[0]

        onnx_model.remap_input_names(input_name_map)
        onnx_model.remove_unused()
        onnx_model.ReInit()
        return onnx_model
