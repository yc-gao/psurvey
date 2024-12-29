#!/usr/bin/env python3
import argparse

from onnxutils.common import OnnxModel
from onnxutils.optim import find_optimizer


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('--optim', action='append', default=[])
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    for x in options.optim:
        optimizer = find_optimizer(x)
        if optimizer is None:
            raise RuntimeError(f"can not find '{x}' optimizer, ignore")
        onnx_model = optimizer.apply(onnx_model)

    if options.output:
        onnx_model.save(options.output)


if __name__ == "__main__":
    main()
