#!/usr/bin/env python3
import argparse

from onnxutils.common import OnnxModel
from onnxutils.optim import DagMatcher

conv_relu_pattern = DagMatcher({
    'op_type': 'Relu',
    'inputs': [
        {
            'op_type': 'Conv'
        }
    ]
})


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)

    dags = conv_relu_pattern.MatchAllDags(onnx_model)
    print(len(dags))


if __name__ == "__main__":
    main()
