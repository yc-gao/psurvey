import onnx
import numpy as np

from ..onnx_model import OnnxModel
from .registry import optimizer


@optimizer('convert-shape-to-initializer')
class ConvertShapeToInitializer:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.transaction() as t:
            for node in onnx_model.get_nodes_by_optype('Shape'):
                vinfo = onnx_model.get_vinfo_by_name(node.input[0])
                if vinfo is None:
                    continue
                if not vinfo.type.HasField('tensor_type'):
                    continue
                shape = [x.dim_value if x.HasField(
                    'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]
                if any(x == -1 for x in shape):
                    continue

                t.add_initializer(onnx.numpy_helper.from_array(
                    np.array(shape, dtype=np.int64),
                    node.output[0]))
                t.remove_node(node)

        return onnx_model
