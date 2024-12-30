#!/usr/bin/env python3
import argparse

from onnxutils.common import OnnxModel
from onnxutils.optim import find_optimizer
from onnxutils.onnx2torch import convert


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    onnx_model = OnnxModel.from_file(options.model)

    optimizers = [
        'convert-constant-to-initializer',
        'convert-shape-to-initializer',
        'onnx-simplifier',
        'convert-shape-to-initializer',
    ]
    for x in optimizers:
        optimizer = find_optimizer(x)
        if optimizer is None:
            raise RuntimeError(f"can not find '{x}' optimizer, ignore")
        onnx_model = optimizer.apply(onnx_model)

    with onnx_model.session() as sess:
        for node in onnx_model.proto().graph.node:
            if node.name == '':
                node.name = sess.unique_name()

    torch_module = convert(onnx_model)


if __name__ == "__main__":
    main()
