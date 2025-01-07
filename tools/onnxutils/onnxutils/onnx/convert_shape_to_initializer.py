import numpy as np

from .onnx_model import OnnxModel
from .onnx_node import OnnxTensor

from .registry import optimizer


@optimizer('convert-shape-to-initializer')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = onnx_model.output_names()
        with onnx_model.session() as sess:
            for node in onnx_model.nodes():
                if node.op_type() != 'Shape':
                    continue
                if node.outputs()[0] in output_names:
                    continue

                vinfo = onnx_model.get_vinfo_by_name(node.inputs()[0])
                if vinfo is None:
                    continue

                shape = [x.dim_value if x.HasField(
                    'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]
                if any(x == -1 for x in shape):
                    continue

                sess.add_initializer(OnnxTensor.from_numpy(
                    np.array(shape, dtype=np.int64), node.outputs()[0]).proto())
                sess.remove_node(node)

        return onnx_model
