#!/usr/bin/env python3
import argparse
import functools
from pathlib import Path
import json
import copy

import numpy as np
import onnx

from onnx_model import OnnxModel
from matcher import DagMatcher

from eliminate_constant import EliminateConstant
from eliminate_cast import EliminateCast
from eliminate_identity import EliminateIdentity
from eliminate_qdq import EliminateQdq

quanzed_gemm_relu_pattern = DagMatcher({
    'id': 0,
    'op_type': 'QuantizeLinear',
    'inputs': [
        {
            'id': 1,
            'op_type': 'Relu',
            'inputs': [
                {
                    'id': 2,
                    'op_type': 'Gemm',
                    'inputs': [
                        {
                            'id': 3,
                            'op_type': 'DequantizeLinear'
                        },
                        {
                            'id': 4,
                            'op_type': 'DequantizeLinear'
                        }
                    ]
                }
            ]
        }
    ]
})
quanzed_gemm_bn_relu_pattern = DagMatcher({
    'id': 0,
    'op_type': 'QuantizeLinear',
    'inputs': [
        {
            'id': 1,
            'op_type': 'Relu',
            'inputs': [
                {
                    'id': 2,
                    'op_type': 'BatchNormalization',
                    'inputs': [
                        {
                            'id': 3,
                            'op_type': 'Gemm',
                            'inputs': [
                                {
                                    'id': 4,
                                    'op_type': 'DequantizeLinear'
                                },
                                {
                                    'id': 5,
                                    'op_type': 'DequantizeLinear'
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
})
quanzed_conv_bn_relu_pattern = DagMatcher({
    'id': 0,
    'op_type': 'QuantizeLinear',
    'inputs': [
        {
            'id': 1,
            'op_type': 'Relu',
            'inputs': [
                {
                    'id': 2,
                    'op_type': 'BatchNormalization',
                    'inputs': [
                        {
                            'id': 3,
                            'op_type': 'Conv',
                            'inputs': [
                                {
                                    'id': 4,
                                    'op_type': 'DequantizeLinear'
                                },
                                {
                                    'id': 5,
                                    'op_type': 'DequantizeLinear'
                                }
                            ]
                        }
                    ]
                }
            ]
        }
    ]
})


def unique_count(items: list) -> dict:
    tmp = {}
    for item in items:
        tmp[item] = tmp.get(item, 0) + 1
    return tmp


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('model')
    return parser.parse_args()


def qdqnode_to_dict(q_node, onnx_model, dtype):
    scale = onnx.numpy_helper.to_array(
        onnx_model.get_initializer_by_name(q_node.input[1]))
    zero_point = onnx.numpy_helper.to_array(
        onnx_model.get_initializer_by_name(q_node.input[2]))
    assert scale.shape == zero_point.shape
    elem_count = functools.reduce(lambda a, b: a * b, scale.shape, 1)

    scale = np.reshape(scale, (elem_count, ))
    zero_point = np.reshape(zero_point, (elem_count, )).astype(np.float32)
    if dtype == 'int8':
        return [
            {
                'bitwidth': 8,
                'dtype': 'int',
                'offset': int(z),
                'scale': float(s)
            } for s, z in zip(scale, zero_point)
        ]
    # qualcomm magic
    zero_point = 65280 * zero_point / 255
    scale = 255 * scale / 65280
    return [
        {
            'bitwidth': 16,
            'dtype': 'float',
            'offset': int(z),
            'scale': float(s)
        } for s, z in zip(scale, zero_point)
    ]


def dag_format(dag):
    dag = copy.deepcopy(dag)
    node = dag['node']
    dag['node'] = {
        'op_type': node.op_type,
        'name': node.name
    }

    for idx, d in enumerate(dag.get('inputs', [])):
        dag['inputs'][idx] = dag_format(d)
    return dag


def main():
    options = parse_options()

    activation_encodings = {}

    encodings_unset = set()

    onnx_model = OnnxModel(options.model)
    onnx_model = EliminateCast.apply(onnx_model)
    onnx_model = EliminateIdentity.apply(onnx_model)
    onnx_model = EliminateConstant.apply(onnx_model)

    unquanzed_model = EliminateQdq.apply(onnx_model.clone())

    for node in reversed(onnx_model.nodes()):
        ret, dag = quanzed_gemm_relu_pattern.Match(node, onnx_model)
        if ret:
            q_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 0)
            relu_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 1)
            gemm_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 2)
            dq_node0 = quanzed_gemm_bn_relu_pattern.FindNode(dag, 3)
            dq_node1 = quanzed_gemm_bn_relu_pattern.FindNode(dag, 4)

            encodings_unset.update(relu_node.output)
            encodings_unset.update(gemm_node.output)

            unquanzed_gemm_node = unquanzed_model.get_node_by_name(
                gemm_node.name)
            activation_encodings[unquanzed_gemm_node.input[0]
                                 ] = qdqnode_to_dict(dq_node0, onnx_model, 'fp16')

            unquanzed_relu_node = unquanzed_model.get_node_by_name(
                relu_node.name)
            activation_encodings[unquanzed_relu_node.output[0]] = qdqnode_to_dict(
                q_node, onnx_model, 'int8')
            continue
        ret, dag = quanzed_gemm_bn_relu_pattern.Match(node, onnx_model)
        if ret:
            q_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 0)
            relu_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 1)
            bn_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 2)
            gemm_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 3)
            dq_node0 = quanzed_gemm_bn_relu_pattern.FindNode(dag, 4)
            dq_node1 = quanzed_gemm_bn_relu_pattern.FindNode(dag, 5)

            encodings_unset.update(relu_node.output)
            encodings_unset.update(bn_node.output)
            encodings_unset.update(gemm_node.output)

            unquanzed_gemm_node = unquanzed_model.get_node_by_name(
                gemm_node.name)
            activation_encodings[unquanzed_gemm_node.input[0]
                                 ] = qdqnode_to_dict(dq_node0, onnx_model, 'fp16')

            unquanzed_relu_node = unquanzed_model.get_node_by_name(
                relu_node.name)
            activation_encodings[unquanzed_relu_node.output[0]] = qdqnode_to_dict(
                q_node, onnx_model, 'int8')
            continue
        ret, dag = quanzed_conv_bn_relu_pattern.Match(node, onnx_model)
        if ret:
            q_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 0)
            relu_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 1)
            bn_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 2)
            conv_node = quanzed_gemm_bn_relu_pattern.FindNode(dag, 3)
            dq_node0 = quanzed_gemm_bn_relu_pattern.FindNode(dag, 4)
            dq_node1 = quanzed_gemm_bn_relu_pattern.FindNode(dag, 5)

            encodings_unset.update(relu_node.output)
            encodings_unset.update(bn_node.output)
            encodings_unset.update(conv_node.output)

            unquanzed_conv_node = unquanzed_model.get_node_by_name(
                conv_node.name)

            activation_encodings[unquanzed_conv_node.input[0]
                                 ] = qdqnode_to_dict(dq_node0, onnx_model, 'fp16')

            unquanzed_relu_node = unquanzed_model.get_node_by_name(
                relu_node.name)
            activation_encodings[unquanzed_relu_node.output[0]] = qdqnode_to_dict(
                q_node, onnx_model, 'int8')
            continue

    final_activation_encodings = {}
    for node in unquanzed_model.nodes():
        for output in node.output:
            if output in encodings_unset:
                continue
            final_activation_encodings[output] = [
                {'bitwidth': 16, 'dtype': 'float'}]
    final_activation_encodings.update(activation_encodings)

    for input_name in unquanzed_model.input_names():
        tmp = final_activation_encodings.get(input_name, None)
        if tmp:
            final_activation_encodings[input_name] = [
                {
                    'bitwidth': 16,
                    'dtype': 'float',
                    'scale': float(t['scale'] * 65280 / 255),
                    'offset': int(t['offset'] * 255 / 65280),
                } if t['dtype'] == 'float' else t for t in tmp]

    onnx_model.topological_sort()
    unquanzed_model.topological_sort()
    if options.output:
        output = Path(options.output)
        output.mkdir(parents=True, exist_ok=True)
        onnx_model.save(output/'model.onnx')
        unquanzed_model.save(output/'model.unquanzed.onnx')
        with open(output/'encodings.json', 'w') as f:
            json.dump({
                'activation_encodings': final_activation_encodings,
                'param_encodings': [],
            }, f)


if __name__ == "__main__":
    main()
