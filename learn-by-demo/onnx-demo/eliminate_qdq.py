from onnx_model import OnnxModel

from matcher import DagMatcher

dqq_matcher = DagMatcher({
    'op_type': 'DequantizeLinear',
    'inputs': [
        {
            'op_type': 'QuantizeLinear',
        }
    ]
})
qdq_matcher = DagMatcher({
    'op_type': 'QuantizeLinear',
    'inputs': [
        {
            'op_type': 'DequantizeLinear',
        }
    ]
})


class EliminateQdq:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        input_name_map = {}

        node_merged = set()
        for node in reversed(onnx_model.nodes()):
            if node.name in node_merged:
                continue
            ret, dag = dqq_matcher.Match(node, onnx_model)
            if ret:
                dq_node = dag['node']
                q_node = dag['inputs'][0]['node']
                if dq_node.name not in node_merged and q_node.name not in node_merged and len(onnx_model.get_nodes_by_input_name(q_node.output[0])) < 2:
                    input_name_map[dq_node.output[0]] = q_node.input[0]
                    node_merged.add(dq_node.name)
                    node_merged.add(q_node.name)
            else:
                ret, dag = qdq_matcher.Match(node, onnx_model)
                if ret:
                    q_node = dag['node']
                    dq_node = dag['inputs'][0]['node']
                    if dq_node.name not in node_merged and q_node.name not in node_merged and len(onnx_model.get_nodes_by_input_name(dq_node.output[0])) < 2:
                        input_name_map[q_node.output[0]] = dq_node.input[0]
                        node_merged.add(dq_node.name)
                        node_merged.add(q_node.name)

        onnx_model.remap_input_names(input_name_map)
        onnx_model.remove_unused()
        onnx_model.ReInit()
        return onnx_model
