import os
import uuid
from collections import Counter

import onnx
from onnx.utils import Extractor
from onnx.onnx_ml_pb2 import ModelProto
from onnx.onnx_ml_pb2 import TensorProto
from onnx.onnx_ml_pb2 import NodeProto

from .onnx_tensor import OnnxTensor
from .onnx_node import OnnxNode


class OnnxModel:
    @staticmethod
    def from_file(fpath):
        if isinstance(fpath, os.PathLike):
            fpath = os.fspath(fpath)
        return OnnxModel(onnx.load(fpath))

    def reindex(self, model: ModelProto):
        model = onnx.shape_inference.infer_shapes(model)
        self._proto = model

        self._nodes = tuple(OnnxNode(x) for x in self._proto.graph.node)
        self._inputs = tuple(x for x in self._proto.graph.input)
        self._outputs = tuple(x for x in self._proto.graph.output)
        self._initializers = tuple(OnnxTensor(x)
                                   for x in self._proto.graph.initializer)

        self._name_to_node = {
            x.name(): x for x in self._nodes
        }
        self._output_to_node = {
            output: node for node in self._nodes for output in node.outputs()
        }

        self._name_to_initializer = {
            x.name(): x for x in self._initializers
        }

        self._name_to_input = {
            x.name: x for x in self._inputs
        }
        self._name_to_output = {
            x.name: x for x in self._outputs
        }
        self._name_to_vinfo = {
            x.name: x for x in self._proto.graph.value_info
        }
        self._name_to_vinfo.update(self._name_to_input)
        self._name_to_vinfo.update(self._name_to_output)

        self._name_to_counter = Counter(
            [input_name for node in self._proto.graph.node for input_name in node.input] +
            [x.name for x in self._proto.graph.output]
        )

    def clone(self):
        t = ModelProto()
        t.CopyFrom(self._proto)
        return OnnxModel(t)

    def save(self, fpath):
        if isinstance(fpath, os.PathLike):
            fpath = os.fspath(fpath)
        onnx.save(self._proto, fpath)

    def __init__(self, model: ModelProto):
        self.reindex(model)

    def proto(self):
        return self._proto

    def opsets(self):
        return tuple(x for x in self._proto.opset_import)

    def inputs(self):
        return self._inputs

    def input_names(self):
        return set({x.name for x in self._inputs})

    def get_input_by_name(self, name):
        return self._name_to_input.get(name, None)

    def outputs(self):
        return self._outputs

    def output_names(self):
        return set({x.name for x in self._outputs})

    def get_output_by_name(self, name):
        return self._name_to_output.get(name, None)

    def initializers(self):
        return self._initializers

    def initializer_names(self):
        return set({x.name for x in self._initializers})

    def get_initializer_by_name(self, name) -> OnnxTensor:
        return self._name_to_initializer.get(name, None)

    def nodes(self):
        return self._nodes

    def node_names(self):
        return set({x.name for x in self._nodes})

    def get_node_by_name(self, name):
        return self._name_to_node.get(name, None)

    def get_node_by_output(self, output):
        return self._output_to_node.get(output, None)

    def get_vinfo_by_name(self, name):
        return self._name_to_vinfo.get(name, None)

    def get_counter_of_tensor(self, name: str):
        return self._name_to_counter[name]

    def get_counter_of_node(self, name_or_node):
        if isinstance(name_or_node, str):
            name_or_node = self._name_to_node.get(name_or_node, None)
        if name_or_node is None:
            return 0
        return max(
            self._name_to_counter[output_value]
            for output_value in name_or_node.outputs())

    def topological_sort(self, is_deterministic=False):
        node_visited = set()
        sorted_nodes = []

        def do_sort(arr):
            if not is_deterministic:
                return arr
            if not isinstance(arr, list):
                arr = [x for x in arr]
            arr.sort()
            return arr

        def dfs(node):
            if node is None:
                return
            if node.name() in node_visited:
                return
            node_visited.add(node.name())
            for input_name in do_sort(node.inputs()):
                dfs(self.get_node_by_output(input_name))
            sorted_nodes.append(node)

        for output_name in do_sort(self.output_names()):
            dfs(self.get_node_by_output(output_name))

        model = self._proto
        model.graph.ClearField("node")
        model.graph.node.extend([x.proto() for x in sorted_nodes])
        self.reindex(model)

    def extract(self, input_names: list[str], output_names: list[str]):
        e = Extractor(self.proto())
        return OnnxModel(e.extract_model(input_names, output_names))

    def session(self):
        class Session:
            def __init__(self, onnx_model: OnnxModel):
                self._onnx_model = onnx_model

                self._counter = 0

                self._remap_node_inputs = {}

                self._initializers_to_remove = []
                self._initializers_to_add = []

                self._nodes_to_remove = []
                self._nodes_to_add = []

                self._outputs_to_add = []

            def unique_name(self):
                while True:
                    name = f"random_{uuid.uuid1()}_{self._counter}"
                    name = name.replace('-', '_')
                    self._counter += 1
                    if self._onnx_model.get_node_by_name(name) is not None:
                        continue
                    if self._onnx_model.get_vinfo_by_name(name) is not None:
                        continue
                    if self._onnx_model.get_initializer_by_name(name) is not None:
                        continue
                    return name

            def add_initializer(self, tensor: TensorProto):
                self._initializers_to_add.append(tensor)

            def add_initializers(self, tensors: list[TensorProto]):
                self._initializers_to_add.extend(tensors)

            def remove_initializer(self, tensor: OnnxTensor):
                self._initializers_to_remove.append(tensor)

            def add_node(self, node: NodeProto):
                self._nodes_to_add.append(node)

            def remove_node(self, node: OnnxNode):
                self._nodes_to_remove.append(node)

            def remove_nodes(self, nodes):
                self._nodes_to_remove.extend(nodes)

            def remap_node_inputs(self, remap):
                self._remap_node_inputs.update(remap)

            def add_output(self, output):
                self._outputs_to_add.append(output)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                onnx_model = self._onnx_model.proto()
                for node in onnx_model.graph.node:
                    for idx in range(len(node.input)):
                        while True:
                            new_value = self._remap_node_inputs.get(
                                node.input[idx], None)
                            if new_value is None:
                                break
                            node.input[idx] = new_value

                for x in self._initializers_to_remove:
                    onnx_model.graph.initializer.remove(x.proto())
                for x in self._nodes_to_remove:
                    onnx_model.graph.node.remove(x.proto())

                onnx_model.graph.initializer.extend(
                    self._initializers_to_add)
                onnx_model.graph.node.extend(self._nodes_to_add)

                onnx_model.graph.output.extend(self._outputs_to_add)

                e = Extractor(onnx_model)
                new_model = e.extract_model(
                    [x.name for x in onnx_model.graph.input],
                    [x.name for x in onnx_model.graph.output])
                self._onnx_model.reindex(new_model)

        return Session(self)
