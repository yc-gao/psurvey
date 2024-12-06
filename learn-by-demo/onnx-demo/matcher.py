from onnx_model import ModelVistor


class NodeMatcher:
    properties = ('op_type', 'name')

    def __init__(self, pattern):
        self.pattern = pattern

    def Match(self, node):
        if not node:
            return False
        for prop in self.properties:
            if prop in self.pattern:
                if isinstance(self.pattern[prop], str):
                    if getattr(node, prop) != self.pattern[prop]:
                        return False
                else:
                    if not self.pattern[prop].Match(getattr(node, prop)):
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
            'id': self.pattern.get('id', -1),
            'node': node,
            'inputs': inputs
        }

    def MatchAll(self, vistor: ModelVistor):
        dags = []
        node_matched = set()
        for node in reversed(vistor.nodes()):
            if node.name in node_matched:
                continue
            ret, dag = self.Match(node, vistor)
            if not ret:
                continue
            if any([node.name in node_matched for node in self.GetNodes(dag)]):
                continue
            node_matched.update([node.name for node in self.GetNodes(dag)])
            dags.append(dag)

        return dags

    def GetNodes(self, dag):
        nodes = []
        if 'node' in dag:
            nodes.append(dag['node'])
        for input in dag.get('inputs', []):
            nodes.extend(self.GetNodes(input))
        return nodes

    def FindNode(self, dag, idnum):
        if dag.get('id', -1) == idnum:
            return dag['node']
        for input in dag.get('inputs', []):
            ret = self.FindNode(input, idnum)
            if ret:
                return ret
        return None
