#!/usr/bin/env python3
import argparse

from onnxutils.common import OnnxModel
from onnxutils.optim import apply_optimizers


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('--optim', action='append', default=[])
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    onnx_model = apply_optimizers(options.optim)

    if options.output:
        onnx_model.save(options.output)


if __name__ == "__main__":
    main()
