#!/usr/bin/env python3
import os
import uuid

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

    def clone(self):
        t = onnx.ModelProto()
        t.CopyFrom(self._proto)
        return OnnxModel(t)

    def save(self, fpath):
        if isinstance(fpath, os.PathLike):
            fpath = os.fspath(fpath)
        onnx.save(self._proto, fpath)

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

    def outputs(self):
        return [x for x in self.graph().output]

    def output_names(self):
        return [x.name for x in self.outputs()]

    def get_output_by_name(self, name):
        return self.name_to_output.get(name, None)

    def initializers(self):
        return [x for x in self.graph().initializer]

    def initializer_names(self):
        return [x.name for x in self.initializers()]

    def get_initializer_by_name(self, name):
        return self.name_to_initializer.get(name, None)

    def vinfos(self):
        return [x for x in self.graph().value_info]

    def vinfo_names(self):
        return [x.name for x in self.vinfos()]

    def get_vinfo_by_name(self, name):
        return self.name_to_vinfo.get(name, None)

    def nodes(self):
        return [x for x in self.graph().node]

    def node_names(self):
        return [x.name for x in self.nodes()]

    def get_node_by_name(self, name):
        return self.name_to_node.get(name, None)

    def get_nodes_by_optype(self, optype):
        return [x for x in self.nodes() if x.op_type == optype]

    def get_node_by_output_name(self, output_name):
        return self.output_name_to_node.get(output_name, None)

    def extract(self, input_names: list[str], output_names: list[str]):
        e = Extractor(self.model())
        return OnnxModel(e.extract_model(input_names, output_names))

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

    def transaction(self):
        class ModelTransaction:
            def __init__(self, onnx_model: OnnxModel):
                self.onnx_model = onnx_model

                self.input_names_remap = {}

                self.initializers_to_remove = []
                self.initializers_to_add = []

                self.nodes_to_add = []
                self.nodes_to_remove = []
                self.nodes_to_remove_by_name = []

                self.counter = 0

            def unique_name(self):
                while True:
                    name = f"random_{uuid.uuid1()}_{self.counter}"
                    self.counter += 1
                    if self.onnx_model.get_node_by_name(name) is not None:
                        continue
                    if self.onnx_model.get_vinfo_by_name(name) is not None:
                        continue
                    if self.onnx_model.get_initializer_by_name(name) is not None:
                        continue
                    return name

            def remove_initializer(self, initializer):
                self.initializers_to_remove.append(initializer)

            def remove_initializers(self, initializers):
                self.initializers_to_remove.extend(initializers)

            def add_initializer(self, initializer):
                self.initializers_to_add.append(initializer)

            def add_initializers(self, initializers):
                self.initializers_to_add.extend(initializers)

            def remove_node(self, node):
                self.nodes_to_remove.append(node)

            def remove_node_by_name(self, name):
                self.nodes_to_remove_by_name.append(name)

            def remove_nodes(self, nodes):
                self.nodes_to_remove.extend(nodes)

            def add_node(self, node):
                self.nodes_to_add.append(node)

            def add_nodes(self, nodes):
                self.nodes_to_add.extend(nodes)

            def remap_input_names(self, remap):
                self.input_names_remap.update(remap)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_value is not None:
                    raise exc_value

                for node in self.onnx_model.nodes():
                    for idx, _ in enumerate(node.input):
                        while True:
                            new_input_name = self.input_names_remap.get(
                                node.input[idx], None)
                            if new_input_name is None:
                                break
                            node.input[idx] = new_input_name

                for initializer in self.initializers_to_remove:
                    self.onnx_model.graph().initializer.remove(initializer)
                for node in self.nodes_to_remove:
                    self.onnx_model.graph().node.remove(node)
                for name in self.nodes_to_remove_by_name:
                    node = self.onnx_model.get_node_by_name(name)
                    if node is not None:
                        self.onnx_model.graph().node.remove(node)

                if self.initializers_to_add:
                    self.onnx_model.graph().initializer.extend(self.initializers_to_add)
                if self.nodes_to_add:
                    self.onnx_model.graph().node.extend(self.nodes_to_add)

                proto_model = self.onnx_model.model()
                e = Extractor(proto_model)
                new_model = e.extract_model(
                    [x.name for x in proto_model.graph.input],
                    [x.name for x in proto_model.graph.output])
                self.onnx_model.reindex(new_model)
                self.onnx_model.topological_sort()

        return ModelTransaction(self)
