import onnx
import numpy as np

from .onnx_model import OnnxModel

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
                new_conv = node.clone().proto()
                new_conv.input[:] = [node.inputs()[0], node.inputs()[1]]
                new_conv.output[0] = sess.unique_name()
                sess.add_node(new_conv)

                scale = onnx.numpy_helper.from_array(
                    np.ones_like(bias),
                    sess.unique_name()
                )
                B = onnx.numpy_helper.from_array(
                    np.zeros_like(bias),
                    sess.unique_name()
                )
                input_mean = onnx.numpy_helper.from_array(
                    -bias,
                    sess.unique_name()
                )
                input_var = onnx.numpy_helper.from_array(
                    np.ones_like(bias) - 1e-5,
                    sess.unique_name()
                )

                sess.add_initializers([scale, B, input_mean, input_var])

                bn_node = onnx.helper.make_node(
                    'BatchNormalization',
                    [new_conv.output[0], scale.name, B.name,
                     input_mean.name, input_var.name],
                    [node.outputs()[0]],
                )
                sess.add_node(bn_node)
        return onnx_model
