from .onnx_model import OnnxModel

from .registry import optimizer


@optimizer('onnx-simplifier')
class OnnxSimplifier:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        from onnxsim.onnx_simplifier import simplify
        from onnxsim.model_info import print_simplifying_info
        model, ret = simplify(onnx_model.proto())
        if ret:
            print_simplifying_info(onnx_model.proto(), model)
            return OnnxModel(model)
        return onnx_model
