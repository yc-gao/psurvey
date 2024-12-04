from onnx_model import ModelVistor


class NodeMatcher:
    properties = ('op_type', 'name')

    def __init__(self, pattern):
        self.pattern = pattern

    def Match(self, node):
        if not node:
            return False
        for prop in self.properties:
            if prop in self.pattern and getattr(node, prop) != self.pattern[prop]:
                return False
        return True


class DagMatcher:
    def __init__(self, pattern):
        self.pattern = pattern

    def Match(self, node, vistor: ModelVistor):
        node_matcher = NodeMatcher(self.pattern)
        if not node_matcher.Match(node):
            return False, None
        ipattern = self.pattern.get('inputs', [])
        inodes = [vistor.get_node_by_output_name(
            x) for x in node.input][:len(ipattern)]
        if len(ipattern) != len(inodes):
            return False, None

        inputs = []
        for p, n in zip(ipattern, inodes):
            dag_matcher = DagMatcher(p)
            ret, dag = dag_matcher.Match(n, vistor)
            if not ret:
                return False, None
            inputs.append(dag)

        return True, {
            'node': node,
            'inputs': inputs
        }
