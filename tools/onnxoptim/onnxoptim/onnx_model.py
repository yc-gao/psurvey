#!/usr/bin/env python3
import os

import onnx
from onnx.utils import Extractor


class OnnxModel:
    def __init__(self, model):
        self.reindex(model)

    def reindex(self, model):
        if isinstance(model, os.PathLike):
            model = os.fspath(model)
        if isinstance(model, str):
            model = onnx.load(model)
        assert isinstance(model, onnx.ModelProto)
        model = onnx.shape_inference.infer_shapes(model)
        self._proto = model

        self.name_to_node = {
            x.name: x for x in self._proto.graph.node
        }
        self.output_name_to_node = {
            output: node for node in self._proto.graph.node for output in node.output
        }
        self.name_to_initializer = {
            x.name: x for x in self._proto.graph.initializer
        }
        self.name_to_input = {
            x.name: x for x in self._proto.graph.input
        }
        self.name_to_output = {
            x.name: x for x in self._proto.graph.output
        }
        self.name_to_vinfo = {
            x.name: x for x in self._proto.graph.value_info
        }
        self.name_to_vinfo.update(self.name_to_input)
        self.name_to_vinfo.update(self.name_to_output)

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

    def get_input_by_name(self, name):
        return self.name_to_input.get(name, None)

    def add_input(self, input):
        self.add_inputs([input])

    def add_inputs(self, inputs):
        self.graph().input.extend(inputs)

    def remove_input(self, input):
        self.graph().input.remove(input)
        self.remove_inputs([input])

    def remove_inputs(self, inputs):
        for x in inputs:
            self.remove_input(x)

    def remove_input_by_name(self, name):
        self.remove_input(self.get_input_by_name(name))

    def outputs(self):
        return [x for x in self.graph().output]

    def output_names(self):
        return [x.name for x in self.outputs()]

    def get_output_by_name(self, name):
        return self.name_to_output.get(name, None)

    def add_output(self, output):
        self.add_outputs([output])

    def add_outputs(self, outputs):
        self.graph().output.extend(outputs)

    def remove_output(self, output):
        self.graph().output.remove(output)

    def remove_outputs(self, outputs):
        for x in outputs:
            self.remove_output(x)

    def remove_output_by_name(self, name):
        self.remove_output(self.get_output_by_name(name))

    def clear_outputs(self):
        self.graph().ClearField('output')

    def initializers(self):
        return [x for x in self.graph().initializer]

    def initializer_names(self):
        return [x.name for x in self.initializers()]

    def get_initializer_by_name(self, name):
        return self.name_to_initializer.get(name, None)

    def add_initializer(self, initializer):
        self.add_initializers([initializer])

    def add_initializers(self, initializers):
        self.graph().initializer.extend(initializers)

    def remove_initializer(self, initializer):
        self.graph().initializer.remove(initializer)

    def remove_initializers(self, initializers):
        for x in initializers:
            self.remove_initializer(x)

    def remove_initializer_by_name(self, name):
        self.remove_initializer(self.get_initializer_by_name(name))

    def vinfos(self):
        return [x for x in self.graph().value_info]

    def get_vinfo_by_name(self, name):
        return self.name_to_vinfo.get(name)

    def remove_vinfo(self, vinfo):
        self.graph().value_info.remove(vinfo)

    def remove_vinfos(self, vinfos):
        for x in vinfos:
            self.remove_vinfo(x)

    def remove_vinfo_by_name(self, name):
        self.remove_vinfo(self.get_vinfo_by_name(name))

    def nodes(self):
        return [x for x in self.graph().node]

    def get_node_by_name(self, name):
        return self.name_to_node.get(name, None)

    def get_node_by_output_name(self, name):
        return self.output_name_to_node.get(name, None)

    def add_node(self, node):
        self.add_nodes([node])

    def add_nodes(self, nodes):
        self.graph().node.extend(nodes)

    def get_nodes_by_optype(self, optype):
        return [x for x in self.nodes() if x.op_type == optype]

    def remove_node(self, node):
        self.graph().node.remove(node)

    def remove_nodes(self, nodes):
        for x in nodes:
            self.remove_node(x)

    def remove_node_by_name(self, name):
        self.remove_node(self.get_node_by_name(name))

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

    def remap_input_names(self, input_name_map):
        for node in self.nodes():
            for idx, _ in enumerate(node.input):
                while True:
                    new_input_name = input_name_map.get(node.input[idx], None)
                    if new_input_name:
                        node.input[idx] = new_input_name
                    else:
                        break

    def extract(self, input_names: list[str], output_names: list[str]):
        e = Extractor(self.model())
        return OnnxModel(e.extract_model(input_names, output_names))

    def Transaction(self):
        class TransactionContext(object):
            def __init__(self, onnx_model):
                self.onnx_model = onnx_model

            def __enter__(self):
                return self.onnx_model

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_value is not None:
                    raise exc_value
                proto_model = self.onnx_model.model()
                e = Extractor(proto_model)
                new_model = e.extract_model(
                    [x.name for x in proto_model.graph.input],
                    [x.name for x in proto_model.graph.output])
                self.onnx_model.reindex(new_model)
                self.onnx_model.topological_sort()

        return TransactionContext(self)
