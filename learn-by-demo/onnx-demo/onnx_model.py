#!/usr/bin/env python3
import argparse

import onnx


class OnnxModel:
    def __init__(self, model):
        if isinstance(model, str):
            model = onnx.load(model)
        assert isinstance(model, onnx.ModelProto)
        self.model = model
        self.infer_shape()

    def infer_shape(self):
        self.model = onnx.shape_inference.infer_shapes(self.model)

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

    def remove_unused(self, node=True, input=True, initializer=True):
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

        if input:
            self.remove_inputs(
                [i for i in self.inputs() if i.name not in input_visited])
        if initializer:
            self.remove_initializers(
                [i for i in self.initializers() if i.name not in input_visited])
        if node:
            self.remove_vinfos(
                [i for i in self.vinfos() if i.name not in vinfo_visited])
            self.remove_nodes([node for node in self.nodes()
                              if node.name not in node_visited])

    def replace_input_of_all_nodes(self, old_name, new_name):
        for node in self.nodes():
            for idx, input_name in node.input:
                if input_name == old_name:
                    node.input[idx] = new_name

    def replace_output_of_all_nodes(self, old_name, new_name):
        for node in self.nodes():
            for idx, input_name in node.output:
                if input_name == old_name:
                    node.output[idx] = new_name

    def remap_input_names(self, input_name_map):
        for node in self.nodes():
            for idx, input_name in enumerate(node.input):
                new_input_name = input_name_map.get(input_name, None)
                if new_input_name:
                    node.input[idx] = new_input_name

    def eliminate_cast(self):
        name_to_dtype = {
            v.name: v.type.tensor_type.elem_type for v in self.vinfos() if v.type.HasField('tensor_type')
        }
        name_to_dtype.update({
            v.name: v.type.tensor_type.elem_type for v in self.inputs() if v.type.HasField('tensor_type')
        })
        name_to_dtype.update({
            v.name: v.type.tensor_type.elem_type for v in self.outputs() if v.type.HasField('tensor_type')
        })
        name_to_dtype.update({
            v.name: v.data_type for v in self.initializers()
        })

        input_name_map = {}
        for node in reversed(self.nodes()):
            if node.op_type == 'Cast':
                itype = name_to_dtype.get(node.input[0], None)
                otype = name_to_dtype.get(node.output[0], None)
                if itype and otype and otype == itype:
                    input_name_map[node.output[0]] = node.input[0]

        self.remap_input_names(input_name_map)
        self.remove_unused()

    def merge_qdq(self):
        output_name_to_node = {
            output: node for node in self.nodes() for output in node.output
        }

        input_name_map = {}

        node_merged = set()
        for node in reversed(self.nodes()):
            if node.name in node_merged:
                continue
            if node.op_type == 'DequantizeLinear':
                inode = output_name_to_node.get(node.input[0], None)
                if inode and inode.op_type == 'QuantizeLinear':
                    input_name_map[node.output[0]] = inode.input[0]
                    node_merged.add(node.name)
                    node_merged.add(inode.name)
            elif node.op_type == 'QuantizeLinear':
                inode = output_name_to_node.get(node.input[0], None)
                if inode and inode.op_type == 'DequantizeLinear':
                    input_name_map[node.output[0]] = inode.input[0]
                    node_merged.add(node.name)
                    node_merged.add(inode.name)

        self.remap_input_names(input_name_map)
        self.remove_unused()

    def constant2initializer(self):
        initializer_added = set()
        node_removed = set()

        for node in self.nodes():
            if node.op_type == 'Constant':
                assert len(node.attribute) == 1
                attr = node.attribute[0]
                if attr.HasField('t'):
                    attr.t.name = attr.name
                    initializer_added.add(attr.t)
                    node_removed.add(node)

        self.remove_nodes(node_removed)
        self.add_initializers(initializer_added)


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
