#!/usr/bin/env python3
import argparse
from pathlib import Path

from onnx_model import OnnxModel
from normlize_model import NormlizeModel
from eliminate_cast import EliminateCast
from eliminate_identity import EliminateIdentity
from merge_qdq import MergeQdq


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


def main():
    options = parse_options()
    model = OnnxModel(options.model)

    model = NormlizeModel.apply(model)
    model = EliminateIdentity.apply(model)
    model = EliminateCast.apply(model)

    tensor_name_to_count = unique_count(
        [input_name for node in model.nodes() for input_name in node.input])
    output_name_to_node = {
        output: node for node in model.nodes() for output in node.output
    }
    input_name_to_q = {
        input_name: node for node in model.get_nodes_by_optype('QuantizeLinear') for input_name in node.input
    }
    input_name_to_dq = {
        input_name: node for node in model.get_nodes_by_optype('DequantizeLinear') for input_name in node.input
    }

    unquanzed_model = MergeQdq.apply(model.clone())

    model.topological_sort()
    if options.output:
        output = Path(options.output)
        output.mkdir(parents=True, exist_ok=True)
        unquanzed_model.save(output/'model.unquanzed.onnx')


if __name__ == "__main__":
    main()
