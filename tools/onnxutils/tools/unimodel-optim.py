#!/usr/bin/env python3
import argparse

import numpy as np
import onnx

from onnxutils.onnx import OnnxModel, apply_optimizers


def verify_model(origin_model, new_model, rtol=1e-4, atol=1e-5):
    import numpy as np
    import onnxruntime as ort

    def random_tensor(node):
        str2dtype = {
            'tensor(float)': np.float32
        }
        return np.random.rand(*node.shape).astype(str2dtype[node.type])

    sess0 = ort.InferenceSession(
        origin_model.proto().SerializeToString(),
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])
    sess1 = ort.InferenceSession(
        new_model.proto().SerializeToString(),
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])

    ret = True
    example_inputs = {x.name: random_tensor(x) for x in sess0.get_inputs()}
    sess0_outputs = sess0.run(None, example_inputs)
    sess1_outputs = sess1.run(None, example_inputs)

    ret = [
        (node.name, np.allclose(output0, output1, rtol=rtol, atol=atol))
        for node, output0, output1 in zip(sess0.get_outputs(), sess0_outputs, sess1_outputs)]
    for name, ok in ret:
        print(f"verify {name}...", ok)
    return all(ret)


def custom_fix0(onnx_model: OnnxModel):
    with onnx_model.session() as sess:
        weight = onnx.helper.make_tensor(
            sess.unique_name(),
            onnx.TensorProto.DataType.FLOAT,
            [3, 3],
            [29, 0, 0,
             0, 29, 0,
             0, 0, 9],
        )
        bias = onnx.helper.make_tensor(
            sess.unique_name(),
            onnx.TensorProto.DataType.FLOAT,
            [3],
            [1, 1, 1],
        )
        sess.add_initializers([weight, bias])

        node = onnx.helper.make_node(
            'Gemm',
            ['/MPPModule/fusion/pose_estimator_list.1/pred_sigma_head/pred_sigma_head.7/Sigmoid_output_0',
                weight.name, bias.name],
            [sess.unique_name()],
            sess.unique_name(),
            alpha=1.,
            beta=1.,
            transB=1,
        )
        sess.add_node(node)
        sess.remap_node_inputs({
            '/MPPModule/fusion/pose_estimator_list.1/pred_sigma_head/pred_sigma_head.7/Sigmoid_output_0': node.output[0],
            '/MPPModule/fusion/Add_5_output_0': '/MPPModule/fusion/Slice_5_output_0',
            '/MPPModule/fusion/Add_3_output_0': '/MPPModule/fusion/Slice_4_output_0',
        })

        node = onnx.helper.make_node(
            'Gemm',
            ['/MPPModule/fusion/pose_estimator_list.0/pred_sigma_head/pred_sigma_head.7/Sigmoid_output_0',
                weight.name, bias.name],
            [sess.unique_name()],
            sess.unique_name(),
            alpha=1.,
            beta=1.,
            transB=1,
        )
        sess.add_node(node)
        sess.remap_node_inputs({
            '/MPPModule/fusion/pose_estimator_list.0/pred_sigma_head/pred_sigma_head.7/Sigmoid_output_0': node.output[0],
            '/MPPModule/fusion/Add_1_output_0': '/MPPModule/fusion/Slice_3_output_0',
            '/MPPModule/fusion/Add_output_0': '/MPPModule/fusion/Slice_2_output_0',
        })

    return onnx_model


def custom_fix1(onnx_model: OnnxModel):
    with onnx_model.session() as sess:
        reshape_param = onnx.helper.make_tensor(
            sess.unique_name(),
            onnx.TensorProto.DataType.INT64,
            [4],
            [1, 144, 1, 1000],
        )
        sess.add_initializer(reshape_param)
        reshape_node = onnx.helper.make_node(
            'Reshape',
            ['/RoutingMaskHead/Flatten_output_0', reshape_param.name],
            [sess.unique_name()],
            sess.unique_name(),
        )
        sess.add_node(reshape_node)

        gemm_weight = onnx_model.get_initializer_by_name(
            'network.RoutingMaskHead.use_passable_head.0.weight').to_numpy().T.reshape(1, 144, 1000, 128)
        gemm_weight = onnx.numpy_helper.from_array(
            gemm_weight, sess.unique_name())
        gemm_bias = onnx_model.get_initializer_by_name(
            'network.RoutingMaskHead.use_passable_head.0.bias').to_numpy()
        gemm_bias = onnx.numpy_helper.from_array(gemm_bias, sess.unique_name())
        sess.add_initializers([gemm_weight, gemm_bias])

        matmul_node = onnx.helper.make_node(
            'MatMul',
            [reshape_node.output[0], gemm_weight.name],
            [sess.unique_name()],
            sess.unique_name(),
        )
        sess.add_node(matmul_node)

        reduce_param = onnx.numpy_helper.from_array(
            np.array([1]), sess.unique_name())
        sess.add_initializer(reduce_param)

        reduce_node = onnx.helper.make_node(
            'ReduceSum',
            [matmul_node.output[0], reduce_param.name],
            [sess.unique_name()],
            sess.unique_name(),
            keepdims=1,
            noop_with_empty_axes=0,
        )
        sess.add_node(reduce_node)

        reshape_param = onnx.helper.make_tensor(
            sess.unique_name(),
            onnx.TensorProto.DataType.INT64,
            [2],
            [1, 128],
        )
        sess.add_initializer(reshape_param)
        reshape_node = onnx.helper.make_node(
            'Reshape',
            [reduce_node.output[0], reshape_param.name],
            [sess.unique_name()],
            sess.unique_name(),
        )
        sess.add_node(reshape_node)

        add_node = onnx.helper.make_node(
            'Add',
            [reshape_node.output[0], gemm_bias.name],
            [sess.unique_name()],
            sess.unique_name()
        )
        sess.add_node(add_node)
        sess.remap_node_inputs({
            '/RoutingMaskHead/use_passable_head/use_passable_head.0/Gemm_output_0': add_node.output[0]
        })
    return onnx_model


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)

    new_onnx = custom_fix0(onnx_model)
    new_onnx = custom_fix1(onnx_model)
    new_onnx.topological_sort()

    onnx_model = apply_optimizers(onnx_model, [
        'convert-constant-to-initializer',
        'fold-constant',
        'convert-shape-to-initializer',
        'fold-constant',
        'onnx-simplifier',
    ])

    verify_model(onnx_model, new_onnx)
    if options.output:
        new_onnx.save(options.output)


if __name__ == "__main__":
    main()
