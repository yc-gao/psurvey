import onnx
import numpy as np

from onnxutils.common import OnnxModel

from .registry import optimizer


@optimizer('split-conv-bias-to-bn')
class FoldConstant:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.session() as sess:
            for node in onnx_model.nodes():
                if node.op_type() != 'Conv':
                    continue
                if len(node.inputs()) < 3:
                    continue

                bias = onnx_model.get_initializer_by_name(node.inputs()[2])
                if bias is None:
                    continue
                bias = bias.to_numpy()

                sess.remove_node(node)
                new_conv = node.proto()
                new_conv.input[:] = [node.inputs()[0], node.inputs()[1]]
                new_conv.output[0] = sess.unique_name()
                sess.add_node(new_conv)

                scale = onnx.helper.make_tensor(
                    sess.unique_name(), np.ones_like(bias))
                B = onnx.helper.make_tensor(
                    sess.unique_name(), np.zeros_like(bias))
                input_mean = onnx.helper.make_tensor(
                    sess.unique_name(), -bias)
                input_var = onnx.helper.make_tensor(
                    sess.unique_name(), np.ones_line(bias) - 1e-5)

                sess.add_initializers([scale, B, input_mean, input_var])

                bn_node = onnx.helper.make_node(
                    'BatchNormalization',
                    [new_conv.output[0], scale.name, B.name,
                     input_mean.name, input_var.name],
                    [sess.unique_name()],
                )
                sess.add_node(bn_node)
                sess.remap_node_inputs({node.outputs()[0]: bn_node.output[0]})
        return onnx_model
