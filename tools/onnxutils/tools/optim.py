#!/usr/bin/env python3
import argparse

from onnxutils.onnx import OnnxModel, apply_optimizers


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('--optim', action='append', default=[])
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    onnx_model = apply_optimizers(onnx_model, options.optim)

    with onnx_model.session() as sess:
        for node in onnx_model.proto().graph.node:
            if node.name == '':
                node.name = sess.unique_name()

    if options.output:
        onnx_model.save(options.output)


if __name__ == "__main__":
    main()
