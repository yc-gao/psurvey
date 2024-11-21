#!/usr/bin/env python3
import argparse
import functools

import onnx


class OnnxModel:
    def __init__(self, model):
        assert isinstance(model, (str, onnx.ModelProto))
        if isinstance(model, str):
            model = onnx.load(model)
        self.model = model

    def graph(self):
        return self.model.graph

    # input functions
    def inputs(self):
        return self.graph().input

    def input_names(self):
        return [i.name for i in self.graph().input]

    def get_input_by_name(self, name):
        return ([i for i in self.graph().input if i.name == name] + [None])[0]

    def add_input(self, i):
        self.add_inputs([i])

    def add_inputs(self, inputs):
        self.graph().input.extend(inputs)

    def remove_input(self, i):
        self.graph().input.remove(i)

    def remove_inputs(self, inputs):
        for i in inputs:
            self.remove_input(i)

    def remove_input_by_name(self, name):
        self.remove_input(self.get_input_by_name(name))
    # input functions end

    # output functions
    def outputs(self):
        return self.graph().output

    def output_names(self):
        return [i.name for i in self.graph().output]

    def get_output_by_name(self, name):
        return ([i for i in self.graph().output if i.name == name] + [None])[0]

    def add_output(self, i):
        self.add_outputs([i])

    def add_outputs(self, outputs):
        self.graph().output.extend(outputs)

    def remove_output(self, i):
        self.graph().output.remove(i)

    def remove_outputs(self, outputs):
        for o in outputs:
            self.remove_output(o)

    def remove_output_by_name(self, name):
        self.remove_output(self.get_output_by_name(name))
    # output functions end

    # initializer functions
    def initializers(self):
        return self.graph().initializer

    def initializer_names(self):
        return [i.name for i in self.graph().initializer]

    def get_initializer_by_name(self, name):
        return ([i for i in self.graph().initializer if i.name == name] + [None])[0]

    def add_initializer(self, i):
        self.add_initializers([i])

    def add_initializers(self, initializers):
        self.graph().initializer.extend(initializers)

    def remove_initializer(self, i):
        self.graph().initializer.remove(i)

    def remove_initializers(self, initializers):
        for i in initializers:
            self.remove_initializer(i)

    def remove_initializer_by_name(self, name):
        self.remove_initializer(self.get_initializer_by_name(name))
    # initializer functions end

    # node functions
    def nodes(self):
        return [node for node in self.graph().node]

    def get_node_by_name(self, name):
        return ([node for node in self.nodes() if node.name == name] + [None])[0]

    def get_node_by_output_name(self, output_name):
        return ([node for node in self.nodes() if output_name in node.output] + [None])[0]

    def get_nodes_by_input_name(self, input_name):
        return [node for node in self.nodes() if input_name in node.input]

    def get_nodes_by_optype(self, optype):
        return [node for node in self.nodes() if node.op_type == optype]

    def add_node(self, node):
        self.add_nodes([node])

    def add_nodes(self, nodes):
        self.graph().node.extend(nodes)

    def remove_node(self, node):
        self.graph().node.remove(node)

    def remove_nodes(self, nodes):
        for node in nodes:
            self.remove_node(node)

    def remove_node_by_name(self, name):
        self.remove_node(self.get_node_by_name(name))

    def clear_nodes(self):
        self.graph().ClearField("node")

    # node functions end

    def remove_inputs_unused(self):
        tmp = set(functools.reduce(lambda a, b: a + b,
                  [list(node.input) for node in self.nodes()]))
        inputs_unused = filter(
            lambda x: x.name not in tmp, self.inputs())
        self.remove_inputs(inputs_unused)

    def remove_initializers_unused(self):
        tmp = set(functools.reduce(lambda a, b: a + b,
                  [list(node.input) for node in self.nodes()]))
        initializers_unused = filter(
            lambda x: x.name not in tmp, self.initializers())
        self.remove_initializers(initializers_unused)

    def topological_sort(self, is_deterministic=False):
        output_name_to_node = {
            output: node for node in self.nodes() for output in node.output
        }

        def do_sort(arr):
            if not is_deterministic:
                return arr
            if not isinstance(arr, list):
                arr = [x for x in arr]
            arr.sort(key=lambda x: x.name if hasattr(x, 'name') else x)
            return arr

        sorted_node_set = set()
        sorted_nodes = []

        def dfs(node):
            for input_name in do_sort(node.input):
                n = output_name_to_node.get(input_name, None)
                if n and (n.name not in sorted_node_set):
                    dfs(n)
            sorted_node_set.add(node.name)
            sorted_nodes.append(node)

        for output_name in do_sort(self.output_names()):
            dfs(output_name_to_node[output_name])
        self.graph().ClearField("node")
        self.graph().node.extend(sorted_nodes)

    @staticmethod
    def replace_input_name_of_node(node, old_input_name, new_input_name):
        assert isinstance(old_input_name, str) and isinstance(
            new_input_name, str)
        for idx, i in enumerate(node.input):
            if i == old_input_name:
                node.input[idx] = new_input_name

    @staticmethod
    def replace_output_name_of_node(node, old_output_name, new_output_name):
        assert isinstance(old_output_name, str) and isinstance(
            new_output_name, str)
        for idx, i in enumerate(node.output):
            if i == old_output_name:
                node.output[idx] = new_output_name

    def replace_input_name_of_allnodes(self, old_input_name, new_input_name):
        for node in self.nodes():
            self.replace_input_name_of_node(
                node, old_input_name, new_input_name)

    def replace_output_name_of_allnodes(self, old_output_name, new_output_name):
        for node in self.nodes():
            self.replace_output_name_of_node(
                node, old_output_name, new_output_name)

    # # WARNING: must topo sorted
    # def get_nodes_as_input_of_node(self, node):
    #     tmp = [node]
    #     input_names_set = set(node.input)
    #     for node in self.nodes()[::-1]:
    #         if input_names_set.intersection(node.output):
    #             input_names_set.update(node.input)
    #             tmp.append(node)
    #     return tmp
    #
    # # WARNING: must topo sorted
    # def output_nodes_of_node(self, node):
    #     tmp = [node]
    #     output_names_set = set(node.output)
    #     for node in self.nodes():
    #         if output_names_set.intersection(node.input):
    #             output_names_set.update(node.output)
    #             tmp.append(node)
    #     return tmp


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    model = OnnxModel(options.model)

    model.topological_sort()
    if options.output:
        onnx.save(model.model, options.output)


if __name__ == "__main__":
    main()
