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
                val = getattr(node, prop)
                if callable(val):
                    val = val()
                maybe_val_or_func = self.pattern[prop]
                if callable(maybe_val_or_func) and not maybe_val_or_func(node):
                    return False
                elif maybe_val_or_func != val:
                    return False
        return True

    def MatchDag(self, onnx_model: OnnxModel, node):
        if not self.MatchNode(node):
            return False, None

        ipattern = self.pattern.get('inputs', [])
        if not ipattern:
            return True, {'id': self.pattern.get('id', -1), 'node': node}

        if ipattern[-1] is None:
            ipattern = ipattern[:-1]
            if len(node.inputs()) != len(ipattern):
                return False, None
        else:
            if len(node.inputs()) < len(ipattern):
                return False, None

        inputs = []
        for p, n in zip(ipattern, node.inputs()[:len(ipattern)]):
            dag_matcher = DagMatcher(p)
            ret, dag = dag_matcher.MatchDag(
                onnx_model,
                onnx_model.get_node_by_output(n))
            if not ret:
                return False, None
            inputs.append(dag)
        return True, {
            'id': self.pattern.get('id', -1),
            'node': node,
            'inputs': inputs
        }

    def MatchAllDags(self, onnx_model: OnnxModel, remove_overlapped=True):
        dags = []
        for node in onnx_model.nodes():
            ret, dag = self.MatchDag(onnx_model, node)
            if ret:
                dags.append(dag)

        if not remove_overlapped:
            return dags

        new_dags = []
        node_matched = set()
        for dag in dags:
            all_nodes = DagMatcher.GetAllNodes(dag)
            if any(node.name in node_matched for node in all_nodes):
                continue
            new_dags.append(dag)
            node_matched.update(
                [node.name for node in all_nodes])
        return new_dags
