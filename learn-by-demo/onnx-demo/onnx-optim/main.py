#!/usr/bin/env python3
import argparse
import warnings

from onnx_model import OnnxModel
from registry import find_optimizer

import EliminateIdentity
import EliminateReshape
import EliminateCast
import ConvertConstantToInitializer
import FoldConstant
import ConvertShapeToInitializer


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('--optimizer', action='append', default=[])
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    onnx_model = OnnxModel(options.model)

    for x in options.optimizer:
        optimizer = find_optimizer(x)
        if optimizer is None:
            warnings.warn(f"can not find '{x}' optimizer, ignore")
            continue
        onnx_model = optimizer.apply(onnx_model)

    if options.output:
        onnx_model.save(options.output)


if __name__ == "__main__":
    main()
