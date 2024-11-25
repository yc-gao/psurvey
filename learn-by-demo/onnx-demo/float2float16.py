#!/usr/bin/env python3
import argparse

import onnx
from onnxconverter_common import float16


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output.onnx')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    model = onnx.load(options.model)
    model_fp16 = float16.convert_float_to_float16(model)
    onnx.save(model_fp16, options.output)


if __name__ == "__main__":
    main()
