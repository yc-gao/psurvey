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

from eliminate_constant import EliminateConstant
from eliminate_cast import EliminateCast
from eliminate_identity import EliminateIdentity
from eliminate_qdq import EliminateQdq
from eliminate_dqq_on_initializer import EliminateDqQOnInitializer
from eliminate_relu import EliminateRelu


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

    onnx_model = OnnxModel(options.model)
    onnx_model = EliminateCast.apply(onnx_model)
    onnx_model = EliminateIdentity.apply(onnx_model)
    onnx_model = EliminateConstant.apply(onnx_model)
    onnx_model = EliminateDqQOnInitializer.apply(onnx_model)
    onnx_model = EliminateRelu.apply(onnx_model)

    onnx_model.topological_sort()
    onnx_model, ret = simplify(onnx_model.model)
    assert ret
    onnx_model = OnnxModel(onnx_model)

    unquanzed_model = EliminateQdq.apply(onnx_model.clone())

    onnx_model.topological_sort()
    unquanzed_model.topological_sort()
    if options.output:
        output = Path(options.output)
        output.mkdir(parents=True, exist_ok=True)
        onnx_model.save(output/'model.onnx')
        unquanzed_model.save(output/'model.unquanzed.onnx')


if __name__ == "__main__":
    main()
