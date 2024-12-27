import onnx

from ..onnx_model import OnnxModel
from ..dag_matcher import DagMatcher
from .registry import optimizer

bn_gemm = DagMatcher({
    'id': 0,
    'op_type': 'BatchNormalization',
    'inputs': [
        {
            'id': 1,
            'op_type': 'Gemm'
        },
    ]
})


@optimizer('fuse-bn-into-gemm')
class EliminateReshape:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        import numpy as np

        output_names = set(onnx_model.output_names())
        output_name_to_degree = {
            output_name: 0
            for node in onnx_model.nodes()
            for output_name in node.output
        }
        for node in onnx_model.nodes():
            for input_name in node.input:
                degree = output_name_to_degree.get(input_name, None)
                if degree is not None:
                    output_name_to_degree[input_name] = degree + 1

        with onnx_model.transaction() as t:
            for dag in bn_gemm.MatchAllDags(onnx_model):
                gemm_node = bn_gemm.GetNode(dag, 1)
                if output_name_to_degree.get(gemm_node.output[0], 0) > 1:
                    continue

                bn_node = bn_gemm.GetNode(dag, 0)
                bn_scale = onnx.numpy_helper.to_array(
                    onnx_model.get_initializer_by_name(bn_node.input[1]))
                bn_bias = onnx.numpy_helper.to_array(
                    onnx_model.get_initializer_by_name(bn_node.input[2]))
                bn_mean = onnx.numpy_helper.to_array(
                    onnx_model.get_initializer_by_name(bn_node.input[3]))
                bn_var = onnx.numpy_helper.to_array(
                    onnx_model.get_initializer_by_name(bn_node.input[4]))

                epsilon = 1e-5
                for attr in bn_node.attribute:
                    if attr.name == 'epsilon':
                        epsilon = attr.f
                trans_b = 0
                for attr in gemm_node.attribute:
                    if attr.name == 'transB':
                        trans_b = attr.i

                gm_weight = onnx.numpy_helper.to_array(
                    onnx_model.get_initializer_by_name(gemm_node.input[1]))
                w_factor = np.diag(bn_scale / np.sqrt(bn_var + epsilon))
                if trans_b:
                    gm_weight = (gm_weight.T @ w_factor).T
                else:
                    gm_weight = gm_weight @ w_factor

                new_weight_name = t.unique_name()
                t.add_initializer(onnx.numpy_helper.from_array(
                    gm_weight, new_weight_name))
                gemm_node.input[1] = new_weight_name

                if len(gemm_node.input) < 3:
                    new_bias = - bn_mean * bn_scale / \
                        np.sqrt(bn_var + epsilon) + bn_bias
                    new_bias_name = t.unique_name()
                    t.add_initializer(onnx.numpy_helper.from_array(
                        new_bias, new_bias_name))
                    gemm_node.input.append(new_bias_name)
                else:
                    gm_bias = onnx.numpy_helper.to_array(
                        onnx_model.get_initializer_by_name(gemm_node.input[2]))
                    new_bias = (gm_bias - bn_mean) * bn_scale / \
                        np.sqrt(bn_var + epsilon) + bn_bias
                    new_bias_name = t.unique_name()
                    t.add_initializer(onnx.numpy_helper.from_array(
                        new_bias, new_bias_name))
                    gemm_node.input[2] = new_bias_name

                if bn_node.output[0] in output_names:
                    gemm_node.output[0] = bn_node.output[0]
                # t.remove_node_by_name(gemm_node.name)
                # t.remove_node_by_name(bn_node.name)

                t.add_node(gemm_node)
                t.remap_input_names(
                    {bn_node.output[0]: gemm_node.output[0]})

        return onnx_model
