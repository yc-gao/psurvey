from types import MappingProxyType
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

        self._nodes = {x.name: OnnxNode(x) for x in self._proto.graph.node}
        self._initializers = {
            x.name: OnnxTensor(x)
            for x in self._proto.graph.initializer
        }
        self._input_values = {x.name: x for x in self._proto.graph.input}
        self._output_values = {x.name: x for x in self._proto.graph.output}

        self._value_infos = {
            x.name: x for x in self._proto.graph.value_info
        }
        self._value_infos.update({
            x.name: x for x in self._proto.graph.output
        })
        self._value_infos.update({
            x.name: x for x in self._proto.graph.input
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

    @property
    def proto(self):
        return self._proto

    @property
    def input_values(self):
        return MappingProxyType(self._input_values)

    @property
    def output_values(self):
        return MappingProxyType(self._output_values)

    @property
    def initializers(self):
        return MappingProxyType(self._initializers)

    @property
    def nodes(self):
        return MappingProxyType(self._nodes)

    @property
    def value_infos(self):
        return MappingProxyType(self._value_infos)

    def session(self):
        class Session:
            def __init__(self, onnx_model: OnnxModel):
                self._onnx_model = onnx_model

                self.initializers_to_remove = []
                self.initializers_to_add = []

                self.nodes_to_remove = []

            def remove_initializer(self, tensor: OnnxTensor):
                self.initializers_to_remove.append(tensor)

            def add_initializer(self, tensor: TensorProto):
                self.initializers_to_add.append(tensor)

            def remove_node(self, node: OnnxNode):
                self.nodes_to_remove.append(node)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_value, traceback):
                if exc_value is not None:
                    raise exc_value

                onnx_model = self._onnx_model.proto

                for x in self.initializers_to_remove:
                    onnx_model.graph.initializer.remove(x.proto)
                if self.initializers_to_add:
                    onnx_model.graph.initializer.extend(
                        self.initializers_to_add)

                for x in self.nodes_to_remove:
                    onnx_model.graph.node.remove(x.proto)

                e = Extractor(onnx_model)
                new_model = e.extract_model(
                    [x.name for x in onnx_model.graph.input],
                    [x.name for x in onnx_model.graph.output])
                self._onnx_model.reindex(new_model)

        return Session(self)
