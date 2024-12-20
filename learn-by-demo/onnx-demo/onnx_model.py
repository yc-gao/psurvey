#!/usr/bin/env python3
import os

import onnx


class OnnxModel:
    def __init__(self, model):
        if isinstance(model, os.PathLike):
            model = os.fspath(model)
        if isinstance(model, str):
            model = onnx.load(model)
        assert isinstance(model, onnx.ModelProto)
        model = onnx.shape_inference.infer_shapes(model)
        self._proto = model

        self.node_name_to_node = {
            x.name: x for x in self._proto.graph.node
        }
        self.output_name_to_node = {
            output: node for node in self._proto.graph.node for output in node.output
        }
        self.input_name_to_vinfo = {
            x.name: x for x in self._proto.graph.input
        }
        self.output_name_to_vinfo = {
            x.name: x for x in self._proto.graph.output
        }
        self.initializer_name_to_initializer = {
            x.name: x for x in self._proto.graph.initializer
        }
        self.name_to_vinfo = {
            x.name: x for x in self._proto.graph.value_info
        }
        self.name_to_vinfo.update({
            x.name: x for x in self._proto.graph.input
        })
        self.name_to_vinfo.update({
            x.name: x for x in self._proto.graph.output
        })

    def save(self, fpath):
        if isinstance(fpath, os.PathLike):
            fpath = os.fspath(fpath)
        onnx.save(self._proto, fpath)

    def clone(self):
        t = onnx.ModelProto()
        t.CopyFrom(self._proto)
        return OnnxModel(t)

    def model(self):
        return self._proto

    def graph(self):
        return self._proto.graph

    def inputs(self):
        return [x for x in self.graph().input]

    def input_names(self):
        return [x.name for x in self.inputs()]

    def remove_input(self, input):
        self.remove_inputs([input])

    def remove_inputs(self, inputs):
        for x in inputs:
            self.graph().input.remove(x)

    def outputs(self):
        return [x for x in self.graph().output]

    def output_names(self):
        return [x.name for x in self.outputs()]

    def add_output(self, output):
        self.add_outputs([output])

    def add_outputs(self, outputs):
        self.graph().output.extend(outputs)

    def remove_output(self, output):
        self.remove_outputs([output])

    def remove_outputs(self, outputs):
        for x in outputs:
            self.graph().output.remove(x)

    def initializers(self):
        return [x for x in self.graph().initializer]

    def add_initializer(self, x):
        self.add_initializers([x])

    def add_initializers(self, initializers):
        self.graph().initializer.extend(initializers)

    def remove_initializer(self, initializer):
        self.remove_initializers([initializer])

    def remove_initializers(self, initializers):
        for x in initializers:
            self.graph().initializer.remove(x)

    def vinfos(self):
        return [x for x in self.graph().value_info]

    def get_vinfo_by_name(self, name):
        return self.name_to_vinfo.get(name)

    def remove_vinfo(self, vinfo):
        self.remove_vinfos([vinfo])

    def remove_vinfos(self, vinfos):
        for x in vinfos:
            self.graph().value_info.remove(x)

    def nodes(self):
        return [x for x in self.graph().node]

    def get_node_by_name(self, name):
        return self.node_name_to_node.get(name, None)

    def get_node_by_output_name(self, name):
        return self.output_name_to_node.get(name, None)

    def get_nodes_by_optype(self, optype):
        return [x for x in self.nodes() if x.op_type == optype]

    def remove_node(self, node):
        self.remove_nodes([node])

    def remove_nodes(self, nodes):
        for x in nodes:
            self.graph().node.remove(x)

    def topological_sort(self, is_deterministic=False):

        node_visited = set()
        sorted_nodes = []

        def do_sort(arr):
            if not is_deterministic:
                return arr
            if not isinstance(arr, list):
                arr = [x for x in arr]
            arr.sort(key=lambda x: x.name if hasattr(x, 'name') else x)
            return arr

        def dfs(node):
            if node is None:
                return
            if node.name in node_visited:
                return
            node_visited.add(node.name)
            for input_name in do_sort(node.input):
                dfs(self.output_name_to_node.get(input_name, None))
            sorted_nodes.append(node)

        for output_name in do_sort(self.output_names()):
            dfs(self.output_name_to_node.get(output_name, None))

        self.graph().ClearField("node")
        self.graph().node.extend(sorted_nodes)

    def remove_unused(self, input=True, initializer=True, node=True):
        input_visited = set()
        initializer_visited = set()
        node_visited = set()
        vinfo_visited = set()

        def dfs(node):
            if node is None:
                return
            if node.name in node_visited:
                return
            node_visited.add(node.name)

            vinfo_visited.update(node.output)
            for input_name in node.input:
                if input_name in self.input_name_to_vinfo:
                    input_visited.add(input_name)
                elif input_name in self.initializer_name_to_initializer:
                    initializer_visited.add(input_name)
                else:
                    dfs(self.output_name_to_node.get(input_name, None))

        for output_name in self.output_names():
            dfs(self.output_name_to_node.get(output_name, None))

        if input:
            self.remove_inputs(
                [i for i in self.inputs() if i.name not in input_visited])
        if initializer:
            self.remove_initializers(
                [i for i in self.initializers() if i.name not in initializer_visited])
        if node:
            self.remove_nodes([node for node in self.nodes()
                              if node.name not in node_visited])
            self.remove_vinfos(
                [i for i in self.vinfos() if i.name not in vinfo_visited])

    def remap_input_names(self, input_name_map):
        for node in self.nodes():
            for idx, input_name in enumerate(node.input):
                new_input_name = input_name_map.get(input_name, None)
                if new_input_name:
                    node.input[idx] = new_input_name
