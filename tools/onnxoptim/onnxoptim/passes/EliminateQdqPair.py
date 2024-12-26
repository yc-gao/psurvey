from ..onnx_model import OnnxModel
from ..dag_matcher import DagMatcher
from .registry import optimizer

qdq_pair = DagMatcher({
    'id': 0,
    'op_type': 'DequantizeLinear',
    'inputs': [
        {
            'id': 1,
            'op_type': 'QuantizeLinear'
        },
    ]
})

dqq_pair = DagMatcher({
    'id': 0,
    'op_type': 'QuantizeLinear',
    'inputs': [
        {
            'id': 1,
            'op_type': 'DequantizeLinear'
        },
    ]
})


@optimizer('eliminate-qdq-pair')
class EliminateCast:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.transaction() as t:
            output_name_to_degree = {
                output_name: 0
                for node in onnx_model.nodes()
                for output_name in node.output
            }
            for node in onnx_model.nodes():
                for input_name in node.input:
                    degree = output_name_to_degree.get(input_name, None)
                    if degree is not None:
                        output_name_to_degree[input_name] = degree + 1

            for dag in qdq_pair.MatchAllDags(onnx_model):
                dqnode = qdq_pair.GetNode(dag, 0)
                qnode = qdq_pair.GetNode(dag, 1)
                if output_name_to_degree.get(qnode.output[0], 0) <= 1:
                    t.remap_input_names({
                        dqnode.output[0]: qnode.input[0]
                    })

        with onnx_model.transaction() as t:
            output_name_to_degree = {
                output_name: 0
                for node in onnx_model.nodes()
                for output_name in node.output
            }
            for node in onnx_model.nodes():
                for input_name in node.input:
                    degree = output_name_to_degree.get(input_name, None)
                    if degree is not None:
                        output_name_to_degree[input_name] = degree + 1

            for dag in dqq_pair.MatchAllDags(onnx_model):
                qnode = dqq_pair.GetNode(dag, 0)
                dqnode = dqq_pair.GetNode(dag, 1)
                if output_name_to_degree.get(dqnode.output[0], 0) <= 1:
                    t.remap_input_names({
                        qnode.output[0]: dqnode.input[0]
                    })

        return onnx_model
