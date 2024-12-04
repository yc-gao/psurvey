#!/usr/bin/env python3
import argparse
from pathlib import Path


from onnx_model import OnnxModel
from matcher import DagMatcher

from eliminate_cast import EliminateCast
from eliminate_identity import EliminateIdentity
from eliminate_qdq import EliminateQdq


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

    onnx_model = OnnxModel(options.model)
    onnx_model = EliminateCast.apply(onnx_model)
    onnx_model = EliminateIdentity.apply(onnx_model)

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
