import onnx

from onnx_model import OnnxModel
from registry import optimizer


@optimizer('convert-shape-to-initializer')
class ConvertShapeToInitializer:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        initializer_added = []
        nodes_to_remove = []
        for node in onnx_model.get_nodes_by_optype('Shape'):
            vinfo = onnx_model.get_vinfo_by_name(node.input[0])
            if vinfo is None:
                continue
            if not vinfo.type.HasField('tensor_type'):
                continue
            tensor_shape = vinfo.type.tensor_type.shape
            shape = [x.dim_value if x.HasField(
                'dim_value') else -1 for x in tensor_shape.dim]
            if any(x == -1 for x in shape):
                continue
            initializer_added.append(
                onnx.helper.make_tensor(
                    node.output[0],
                    onnx.TensorProto.DataType.INT64,
                    [len(shape)],
                    shape))
            nodes_to_remove.append(node)

        onnx_model.add_initializers(initializer_added)
        onnx_model.remove_nodes(nodes_to_remove)
        return onnx_model.finalize()
