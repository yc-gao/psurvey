#!/usr/bin/env python3
import argparse
import functools
from pathlib import Path
import json
import copy

import numpy as np
import onnx
from onnxsim.onnx_simplifier import simplify

from onnx_model import OnnxModel

from matcher import DagMatcher
from eliminate_constant import EliminateConstant
from eliminate_cast import EliminateCast
from eliminate_identity import EliminateIdentity
from eliminate_qdq import EliminateQdq
from eliminate_dqq_on_initializer import EliminateDqQOnInitializer
from eliminate_relu import EliminateRelu
from merge_gemm_bn import MergeGemmBN


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


def merge_conv_bn(onnx_model):
    onnx_model.topological_sort()
    # merge conv bn
    onnx_model, ret = simplify(onnx_model.model, perform_optimization=False)
    assert ret
    onnx_model = OnnxModel(onnx_model)
    return onnx_model


def remap_relu_flow(onnx_model):
    pattern = DagMatcher({
        'id': 1,
        'op_type': 'DequantizeLinear',
        'inputs': [
            {
                'id': 2,
                'op_type': 'QuantizeLinear',
                'inputs': [
                    {
                        'id': 3,
                        'op_type': 'Relu'
                    }
                ]
            }
        ]
    })
    input_name_map = {}
    dags = pattern.MatchAll(onnx_model)
    for dag in dags:
        relu = pattern.FindNode(dag, 3)
        dq = pattern.FindNode(dag, 1)
        input_name_map[relu.output[0]] = dq.output[0]

    for node in onnx_model.nodes():
        if node.op_type in ('QuantizeLinear', 'DequantizeLinear'):
            continue
        for idx, input_name in enumerate(node.input):
            new_input_name = input_name_map.get(input_name, None)
            if new_input_name:
                node.input[idx] = new_input_name

    onnx_model.remove_unused()
    onnx_model.ReInit()
    return onnx_model


def merge_dqq_on_same(onnx_model):
    pattern = DagMatcher({
        'id': 1,
        'op_type': 'DequantizeLinear',
        'inputs': [
            {
                'id': 2,
                'op_type': 'QuantizeLinear',
            }
        ]
    })
    union_set = {}
    dags = pattern.MatchAll(onnx_model)
    for dag in dags:
        q_node = pattern.FindNode(dag, 1)
        dq_node = pattern.FindNode(dag, 2)
        if dq_node.input[0] not in union_set:
            union_set[dq_node.input[0]] = [(q_node, dq_node)]
        else:
            union_set[dq_node.input[0]].append((q_node, dq_node))
    input_name_map = {}
    for _, v in union_set.items():
        if len(v) > 1:
            for qdq_pair in v[1:]:
                input_name_map[qdq_pair[1].output[0]] = v[0][1].output[0]

    onnx_model.remap_input_names(input_name_map)
    onnx_model.remove_unused()
    onnx_model.ReInit()
    return onnx_model


def qnode_to_encodings(onnx_model, q_node, dtype='int8'):
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

    zero_point = zero_point * 65280 / 255
    scale = scale * 255 / 65280
    return [
        {
            'bitwidth': 16,
            'dtype': 'float',
            'offset': int(z),
            'scale': float(s)
        } for s, z in zip(scale, zero_point)
    ]


q_conv_dq_pattern = DagMatcher({
    'id': 1,
    'op_type': 'QuantizeLinear',
    'inputs': [
        {
            'id': 2,
            'op_type': 'Conv',
            'inputs': [
                {
                    'id': 3,
                    'op_type': 'DequantizeLinear'
                }
            ]
        }
    ]
})
q_gemm_dq_pattern = DagMatcher({
    'id': 1,
    'op_type': 'QuantizeLinear',
    'inputs': [
        {
            'id': 2,
            'op_type': 'Gemm',
            'inputs': [
                {
                    'id': 3,
                    'op_type': 'DequantizeLinear'
                }
            ]
        }
    ]
})


def main():
    options = parse_options()

    onnx_model = OnnxModel(options.model)
    onnx_model = EliminateCast.apply(onnx_model)
    onnx_model = EliminateIdentity.apply(onnx_model)
    onnx_model = EliminateConstant.apply(onnx_model)
    onnx_model = EliminateDqQOnInitializer.apply(onnx_model)

    onnx_model = merge_dqq_on_same(onnx_model)
    onnx_model = remap_relu_flow(onnx_model)
    onnx_model = EliminateRelu.apply(onnx_model)

    onnx_model = MergeGemmBN.apply(onnx_model)
    onnx_model = merge_conv_bn(onnx_model)

    activation_encodings = {}
    for dag in q_conv_dq_pattern.MatchAll(onnx_model):
        q_node = q_conv_dq_pattern.FindNode(dag, 1)
        conv_node = q_conv_dq_pattern.FindNode(dag, 2)
        dq_node = q_conv_dq_pattern.FindNode(dag, 3)
        unquanzed_conv = unquanzed_model.get_node_by_name(conv_node.name)

        activation_encodings[unquanzed_conv.output[0]
                             ] = qnode_to_encodings(onnx_model, q_node)
        if unquanzed_conv.input[0] not in activation_encodings:
            activation_encodings[unquanzed_conv.input[0]
                                 ] = qnode_to_encodings(onnx_model, dq_node, 'float16')

    for dag in q_gemm_dq_pattern.MatchAll(onnx_model):
        q_node = q_conv_dq_pattern.FindNode(dag, 1)
        gemm_node = q_conv_dq_pattern.FindNode(dag, 2)
        dq_node = q_conv_dq_pattern.FindNode(dag, 3)
        unquanzed_gemm = unquanzed_model.get_node_by_name(gemm_node.name)

        activation_encodings[unquanzed_gemm.output[0]
                             ] = qnode_to_encodings(onnx_model, q_node)
        if unquanzed_gemm.input[0] not in activation_encodings:
            activation_encodings[unquanzed_gemm.input[0]
                                 ] = qnode_to_encodings(onnx_model, dq_node, 'float16')

    for node in unquanzed_model.nodes():
        for output in node.output:
            if output not in activation_encodings:
                activation_encodings[output] = [{
                    'bitwidth': 16,
                    'dtype': 'float',
                }]

    onnx_model.topological_sort()
    unquanzed_model = EliminateQdq.apply(onnx_model.clone())
    unquanzed_model.topological_sort()
    if options.output:
        output = Path(options.output)
        output.mkdir(parents=True, exist_ok=True)
        onnx_model.save(output/'model.qdq.onnx')
        unquanzed_model.save(output/'model.unquanzed.onnx')
        with open(output/'encodings.json', 'w') as f:
            json.dump({'activation_encodings': activation_encodings,
                      'param_encodings': {}}, f)


if __name__ == "__main__":
    main()
