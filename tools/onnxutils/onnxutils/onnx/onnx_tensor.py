import numpy as np
import onnx
from onnx.onnx_ml_pb2 import TensorProto
import torch


class OnnxTensor:
    @classmethod
    def from_numpy(cls, array: np.ndarray, name: str = None):
        onnx_tensor_proto = onnx.numpy_helper.from_array(array, name=name)
        return cls(onnx_tensor_proto)

    @classmethod
    def from_torch(cls, tensor: torch.Tensor, name: str = None):
        array = tensor.detach().cpu().numpy()
        return cls.from_numpy(array, name=name)

    def __init__(self, onnx_tensor: TensorProto):
        self._proto = onnx_tensor

    def proto(self):
        return self._proto

    def name(self) -> str:
        return self._proto.name

    def to_numpy(self) -> np.ndarray:
        return onnx.numpy_helper.to_array(self._proto)

    def to_torch(self) -> torch.Tensor:
        return torch.from_numpy(self.to_numpy().copy())
