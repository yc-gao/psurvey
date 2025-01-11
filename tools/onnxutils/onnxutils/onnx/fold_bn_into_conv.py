import onnx
import numpy as np

from .onnx_model import OnnxModel

from .registry import optimizer
from .dag_matcher import DagMatcher

dag_pattern = DagMatcher({
    'id': 0,
    'op_type': 'BatchNormalization',
    'inputs': [
        {
            'id': 1,
            'op_type': 'Conv'
        },
    ]
})


@optimizer('fold-bn-into-conv')
class FoldConstant:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.session() as sess:
            for dag in dag_pattern.MatchAllDags(onnx_model):
                bn_node = dag_pattern.GetNode(dag, 0)
                conv_node = dag_pattern.GetNode(dag, 1)
                assert bn_node is not None
                assert conv_node is not None

                if onnx_model.get_counter_of_node(conv_node) > 1:
                    continue

                epsilon = bn_node.attributes().get('epsilon', 1e-5)

                bn_scale = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[1]).to_numpy()
                bn_bias = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[2]).to_numpy()
                bn_mean = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[3]).to_numpy()
                bn_var = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[4]).to_numpy()

                weight_factor = bn_scale / np.sqrt(bn_var + epsilon)
                bias_factor = bn_bias - bn_scale * \
                    bn_mean / np.sqrt(bn_var + epsilon)

                conv_weight = onnx_model.get_initializer_by_name(
                    conv_node.inputs()[1])
                sess.remove_initializer(conv_weight)
                conv_weight = conv_weight.to_numpy()
                conv_weight = conv_weight * \
                    weight_factor.reshape(
                        *weight_factor.shape,
                        *([1] * (conv_weight.ndim - weight_factor.ndim))
                    )
                sess.add_initializer(onnx.numpy_helper.from_array(
                    conv_weight,
                    conv_node.inputs()[1]))

                new_conv_node = conv_node.clone().proto()
                new_conv_node.output[0] = bn_node.outputs()[0]

                if len(conv_node.inputs()) >= 3:
                    conv_bias = onnx_model.get_initializer_by_name(
                        new_conv_node.input[2])
                    sess.remove_initializer(conv_bias)

                    conv_bias = conv_bias.to_numpy() + bias_factor
                    sess.add_initializer(
                        onnx.numpy_helper.from_array(
                            conv_bias,
                            conv_node.inputs()[2]
                        )
                    )
                else:
                    conv_bias = bias_factor
                    new_conv_node.input.append(sess.unique_name())
                    sess.add_initializer(
                        onnx.numpy_helper.from_array(
                            conv_bias,
                            new_conv_node.input[2]
                        )
                    )

                sess.remove_node(bn_node)
                sess.remove_node(conv_node)
                sess.add_node(new_conv_node)

            return onnx_model
