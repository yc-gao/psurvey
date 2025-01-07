import os
import warnings
import tempfile


from onnxutils.onnx import OnnxModel

from .registry import optimizer


@optimizer('fold-bn-into-conv')
class FoldConstant:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with tempfile.TemporaryDirectory() as workdir:
            origin_model = os.path.join(workdir, 'origin.onnx')
            output_model = os.path.join(workdir, 'output.onnx')
            onnx_model.save(origin_model)
            if not os.system(
                    f"python3 -m onnxoptimizer {origin_model} {output_model}"):
                return OnnxModel.from_file(output_model)
            warnings.warn("'fuse-bn-into-conv' run failed")
            return onnx_model
