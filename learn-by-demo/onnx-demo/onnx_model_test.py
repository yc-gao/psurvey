#!/usr/bin/env python3
import argparse

import onnx

from onnx_model import OnnxModel


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    model = OnnxModel(options.model)

    model.topological_sort()
    if options.output:
        onnx.save(model.model, options.output)


if __name__ == "__main__":
    main()
