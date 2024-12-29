import numpy as np

from onnxutils.common import OnnxModel, OnnxTensor

from .registry import optimizer


@optimizer('convert-shape-to-initializer')
class _:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_names = set(onnx_model.output_values.keys())
        with onnx_model.session() as sess:
            for node in onnx_model.nodes.values():
                if node.op_type != 'Shape':
                    continue
                if node.output_values[0] in output_names:
                    continue

                vinfo = onnx_model.value_infos.get(node.input_values[0])
                if vinfo is None:
                    continue

                shape = [x.dim_value if x.HasField(
                    'dim_value') else -1 for x in vinfo.type.tensor_type.shape.dim]
                if any(x == -1 for x in shape):
                    continue

                sess.add_initializer(OnnxTensor.from_numpy(
                    np.array(shape, dtype=np.int64), node.output_values[0]).proto)
                sess.remove_node(node)

        return onnx_model
