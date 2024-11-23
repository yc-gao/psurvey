#!/usr/bin/env python3
import argparse
import functools

import onnx


class OnnxModel:
    def __init__(self, model):
        if isinstance(model, str):
            model = onnx.load(model)
        assert isinstance(model, onnx.ModelProto)
        model = onnx.shape_inference.infer_shapes(model)
        self.model = model

    def graph(self):
        return self.model.graph

    # input functions
    def inputs(self):
        return self.graph().input

    def input_names(self):
        return [i.name for i in self.inputs()]

    def get_input_by_name(self, name):
        return ([i for i in self.inputs() if i.name == name] + [None])[0]

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
        return [i.name for i in self.outputs()]

    def get_output_by_name(self, name):
        return ([i for i in self.outputs() if i.name == name] + [None])[0]

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
        return [i.name for i in self.initializers()]

    def get_initializer_by_name(self, name):
        return ([i for i in self.initializers() if i.name == name] + [None])[0]

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

    # vinfo functions
    def vinfos(self):
        return self.graph().value_info

    def vinfo_names(self):
        return [i.name for i in self.vinfos()]

    def get_vinfo_by_name(self, name):
        return ([i for i in self.vinfos() if i.name == name] + [None])[0]

    def add_vinfo(self, i):
        self.add_vinfos([i])

    def add_vinfos(self, vinfos):
        self.graph().vinfo.extend(vinfos)

    def remove_vinfo(self, i):
        self.vinfos().remove(i)

    def remove_vinfos(self, vinfos):
        for i in vinfos:
            self.remove_vinfo(i)

    def remove_vinfo_by_name(self, name):
        self.remove_vinfo(self.get_vinfo_by_name(name))
    # vinfo functions end

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

        node_visited = set()
        sorted_nodes = []

        def dfs(node):
            if node.name in node_visited:
                return
            node_visited.add(node.name)
            for input_name in do_sort(node.input):
                n = output_name_to_node.get(input_name, None)
                if n:
                    dfs(n)
            sorted_nodes.append(node)

        for output_name in do_sort(self.output_names()):
            dfs(output_name_to_node[output_name])
        self.graph().ClearField("node")
        self.graph().node.extend(sorted_nodes)

    def remove_unused(self):
        output_name_to_node = {
            output: node for node in self.nodes() for output in node.output
        }

        node_visited = set()
        vinfo_visited = set()
        input_visited = set()

        def dfs(node):
            if node.name in node_visited:
                return
            node_visited.add(node.name)
            vinfo_visited.update(node.output)
            for input_name in node.input:
                n = output_name_to_node.get(input_name, None)
                if n:
                    dfs(n)
                else:
                    input_visited.add(input_name)
        for output_name in self.output_names():
            dfs(output_name_to_node[output_name])

        self.remove_vinfos(
            [i for i in self.vinfos() if i.name not in vinfo_visited])
        self.remove_nodes([node for node in self.nodes()
                          if node.name not in node_visited])
        self.remove_inputs(
            [i for i in self.inputs() if i.name not in input_visited])
        self.remove_initializers(
            [i for i in self.initializers() if i.name not in input_visited])

    def remap_names(self, io_maps):
        for i in self.inputs():
            new_input_name = io_maps.get(i.name, None)
            if new_input_name:
                i.name = new_input_name
        for i in self.initializers():
            new_input_name = io_maps.get(i.name, None)
            if new_input_name:
                i.name = new_input_name
        for node in self.nodes():
            for idx, input_name in enumerate(node.input):
                new_input_name = io_maps.get(input_name, None)
                if new_input_name:
                    node.input[idx] = new_input_name
            for idx, output_name in enumerate(node.output):
                new_output_name = io_maps.get(output_name, None)
                if new_output_name:
                    node.output[idx] = new_output_name
        for vinfo in self.vinfos():
            new_name = io_maps.get(vinfo.name, None)
            if new_name:
                vinfo.name = new_name


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
