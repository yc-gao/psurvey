import os

import onnx
from onnx.onnx_ml_pb2 import ModelProto


class OnnxModel:
    @staticmethod
    def from_file(fpath):
        if isinstance(fpath, os.PathLike):
            fpath = os.fspath(fpath)
        return OnnxModel(onnx.load(fpath))

    def reindex(self, model: ModelProto):
        self._proto = model

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
