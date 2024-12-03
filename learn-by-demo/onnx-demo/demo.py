#!/usr/bin/env python3
import argparse

from onnx_model import OnnxModel
from normlize_model import NormlizeModel
from eliminate_cast import EliminateCast
from eliminate_identity import EliminateIdentity
from merge_qdq import MergeQdq


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
    model = MergeQdq.apply(model)

    model.topological_sort()
    if options.output:
        model.save(options.output)


if __name__ == "__main__":
    main()
