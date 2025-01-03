from .onnx_model import OnnxModel

# {
#     'op_type': 'QuantizeLinear',
#     'inputs': [
#         {
#             'op_type': 'DequantizeLinear'
#         }
#     ]
# }


class DagMatcher:
    def __init__(self, p={}):
        self.pattern = p

    @staticmethod
    def GetNode(dag, idnum):
        if dag.get('id', -1) == idnum:
            return dag.get('node', None)
        for input in dag.get('inputs', []):
            ret = DagMatcher.GetNode(input, idnum)
            if ret is not None:
                return ret
        return None

    @staticmethod
    def GetAllNodes(dag):
        nodes = []
        if 'node' in dag:
            nodes.append(dag['node'])
        for x in dag.get('inputs', []):
            nodes.extend(DagMatcher.GetAllNodes(x))
        return nodes

    def MatchNode(self, node):
        properties = ('op_type', 'name')
        for prop in properties:
            if prop in self.pattern:
                maybe_val_or_func = self.pattern[prop]
                if callable(maybe_val_or_func) and not maybe_val_or_func(node):
                    return False
                elif maybe_val_or_func != getattr(node, prop):
                    return False
        return True

    def MatchDag(self, onnx_model: OnnxModel, node):
        if not self.MatchNode(node):
            return False, None

        if not self.pattern.get('inputs', None):
            return True, {'id': self.pattern.get('id', -1), 'node': node}
        ipattern = self.pattern['inputs']
        if ipattern[-1] is None:
            if len(node.input) != len(ipattern) - 1:
                return False, None
            ipattern = ipattern[:-1]
        else:
            if len(node.input) < len(ipattern):
                return False, None

        inputs = []
        for p, n in zip(ipattern, node.input[:len(ipattern)]):
            dag_matcher = DagMatcher(p)
            ret, dag = dag_matcher.MatchDag(
                onnx_model,
                onnx_model.get_node_by_output_name(n))
            if not ret:
                return False, None
            inputs.append(dag)
        return True, {
            'id': self.pattern.get('id', -1),
            'node': node,
            'inputs': inputs
        }

    def MatchAllDags(self, onnx_model: OnnxModel):
        dags = []
        node_matched = set()
        for node in reversed(onnx_model.nodes()):
            if node.name in node_matched:
                continue
            ret, dag = self.MatchDag(onnx_model, node)
            if not ret:
                continue
            all_nodes = DagMatcher.GetAllNodes(dag)
            if any(node.name in node_matched for node in all_nodes):
                continue
            dags.append(dag)
            node_matched.update(
                [node.name for node in all_nodes])
        return dags
