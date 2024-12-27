import os
import tempfile
import warnings

from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('fuse-bn-into-conv')
class EliminateReshape:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with tempfile.TemporaryDirectory() as workdir:
            origin_model = os.path.join(workdir, 'origin.onnx')
            output_model = os.path.join(workdir, 'output.onnx')
            onnx_model.save(origin_model)
            if not os.system(
                    f"python3 -m onnxoptimizer {origin_model} {output_model}"):
                return OnnxModel(output_model)
            warnings.warn(f"'fuse-bn-into-conv' run failed")
            return onnx_model
