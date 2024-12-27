#!/usr/bin/env python3
import argparse
import warnings

from onnxoptim import OnnxModel, find_optimizer


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('--optim', action='append', default=[])
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    origin_model = OnnxModel(options.model)

    onnx_model = origin_model.clone()
    for x in options.optim:
        optimizer = find_optimizer(x)
        if optimizer is None:
            warnings.warn(f"can not find '{x}' optimizer, ignore")
            continue
        onnx_model = optimizer.apply(onnx_model)

    if options.output:
        onnx_model.save(options.output)


if __name__ == "__main__":
    main()
