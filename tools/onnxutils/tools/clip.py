#!/usr/bin/env python3
import sys
import argparse
import logging
from pathlib import Path

import onnx

logger = logging.getLogger(__name__)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('--inode', dest='inodes', type=str,
                        default=[], action='append')
    parser.add_argument('--onode', dest='onodes', type=str,
                        default=[], action='append')
    parser.add_argument('model', type=str)
    return parser.parse_args(args)


def main():
    options = parse_args()
    logging.basicConfig(level=logging.INFO)

    model = onnx.load(options.model)
    inodes = options.inodes
    if not inodes:
        inodes = [x.name for x in model.graph.input]
    onodes = options.onodes
    if not onodes:
        onodes = [x.name for x in model.graph.output]

    e = onnx.utils.Extractor(model)
    extracted_model = e.extract_model(inodes, onodes)

    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(extracted_model, output)


if __name__ == '__main__':
    main()
