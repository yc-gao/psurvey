import os

import onnx
from onnx.utils import Extractor
from onnx.onnx_ml_pb2 import ModelProto
from onnx.onnx_ml_pb2 import TensorProto

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

        self._nodes = (OnnxNode(x) for x in self._proto.graph.node)
        self._input_values = (x for x in self._proto.graph.input)
        self._output_values = (x for x in self._proto.graph.output)
        self._initializers = (OnnxTensor(x)
                              for x in self._proto.graph.initializer)

        self._name_to_node = {
            x.name: x for x in self._nodes
        }

        self._name_to_vinfo = {
            x.name: x for x in self._proto.graph.value_info
        }
        self._name_to_vinfo.update({
            x.name: x for x in self._proto.graph.input
        })
        self._name_to_vinfo.update({
            x.name: x for x in self._proto.graph.output
        })

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

    def input_values(self):
        return self._input_values

    def input_names(self):
        return {x.name for x in self._input_values}

    def output_values(self):
        return self._output_values

    def output_names(self):
        return {x.name for x in self._output_values}

    def initializers(self):
        return self._initializers

    def initializer_names(self):
        return {x.name for x in self._initializers}

    def nodes(self):
        return self._nodes

    def node_names(self):
        return {x.name for x in self._nodes}

    def get_node_by_name(self, name):
        return self._name_to_node.get(name, None)

    def get_vinfo_by_name(self, name):
        return self._name_to_vinfo.get(name, None)

    def session(self):
        class Session:
            def __init__(self, onnx_model: OnnxModel):
                self._onnx_model = onnx_model

                self._remap_input_values = {}

                self._initializers_to_remove = []
                self._initializers_to_add = []

                self._nodes_to_remove = []

            def add_initializer(self, tensor: TensorProto):
                self._initializers_to_add.append(tensor)

            def remove_initializer(self, tensor: OnnxTensor):
                self._initializers_to_remove.append(tensor)

            def remove_node(self, node: OnnxNode):
                self._nodes_to_remove.append(node)

            def remap_input_values(self, remap):
                self._remap_input_values.update(remap)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_value is not None:
                    raise exc_value

                onnx_model = self._onnx_model.proto()
                for node in onnx_model.graph.node:
                    for idx in range(len(node.input)):
                        while True:
                            new_value = self._remap_input_values.get(
                                node.input[idx], None)
                            if new_value is None:
                                break
                            node.input[idx] = new_value

                for x in self._initializers_to_remove:
                    onnx_model.graph.initializer.remove(x.proto)
                if self._initializers_to_add:
                    onnx_model.graph.initializer.extend(
                        self._initializers_to_add)

                for x in self._nodes_to_remove:
                    onnx_model.graph.node.remove(x.proto)

                e = Extractor(onnx_model)
                new_model = e.extract_model(
                    [x.name for x in onnx_model.graph.input],
                    [x.name for x in onnx_model.graph.output])
                self._onnx_model.reindex(new_model)

        return Session(self)
