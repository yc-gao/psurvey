import numpy as np

from .onnx_model import OnnxModel
from .onnx_tensor import OnnxTensor

from .registry import optimizer
from .dag_matcher import DagMatcher

dag_pattern = DagMatcher({
    'id': 0,
    'op_type': 'BatchNormalization',
    'inputs': [
        {
            'id': 1,
            'op_type': 'Gemm'
        },
    ]
})


@optimizer('fold-bn-into-gemm')
class FoldConstant:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        with onnx_model.session() as sess:
            for dag in dag_pattern.MatchAllDags(onnx_model):
                bn_node = dag_pattern.GetNode(dag, 0)
                gemm_node = dag_pattern.GetNode(dag, 1)
                assert bn_node is not None
                assert gemm_node is not None

                if onnx_model.get_counter_of_node(gemm_node) > 1:
                    continue

                bn_scale = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[1]).to_numpy()
                bn_bias = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[2]).to_numpy()
                bn_mean = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[3]).to_numpy()
                bn_var = onnx_model.get_initializer_by_name(
                    bn_node.inputs()[4]).to_numpy()

                gm_weight = onnx_model.get_initializer_by_name(
                    gemm_node.inputs()[1]).to_numpy()

                new_gemm_node = gemm_node.clone().proto()

                epsilon = bn_node.attributes().get('epsilon', 1e-5)
                trans_b = gemm_node.attributes().get('transB', 0)

                if trans_b:
                    w_factor = np.diag(bn_scale / np.sqrt(bn_var + epsilon))
                    gm_weight = (gm_weight.T @ w_factor).T
                else:
                    w_factor = np.diag(bn_scale / np.sqrt(bn_var + epsilon))
                    gm_weight = gm_weight @ w_factor

                new_weight_name = sess.unique_name()
                sess.add_initializer(OnnxTensor.from_numpy(
                    gm_weight, new_weight_name).proto())
                new_gemm_node.input[1] = new_weight_name

                if len(new_gemm_node.input) < 3:
                    new_bias = - bn_mean * bn_scale / \
                        np.sqrt(bn_var + epsilon) + bn_bias
                    new_bias_name = sess.unique_name()
                    sess.add_initializer(OnnxTensor.from_numpy(
                        new_bias, new_bias_name).proto())
                    new_gemm_node.input.append(new_bias_name)
                else:
                    gm_bias = onnx_model.get_initializer_by_name(
                        new_gemm_node.input[2]).to_numpy()
                    new_bias = (gm_bias - bn_mean) * bn_scale / \
                        np.sqrt(bn_var + epsilon) + bn_bias
                    new_bias_name = sess.unique_name()
                    sess.add_initializer(OnnxTensor.from_numpy(
                        new_bias, new_bias_name).proto())
                    new_gemm_node.input[2] = new_bias_name

                new_gemm_node.output[0] = bn_node.outputs()[0]

                sess.remove_node(bn_node)
                sess.remove_node(gemm_node)
                sess.add_node(new_gemm_node)

        return onnx_model
