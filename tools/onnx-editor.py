#!/usr/bin/env python3
import os
import sys
import argparse
import logging

import onnx

logger = logging.getLogger(__name__)


def parse_args(args=sys.argv[1:]):
    parser = argparse.ArgumentParser()
    parser.add_argument('--action', choices=['split'], default='split')
    parser.add_argument('-o', '--output', type=str, required=True)
    parser.add_argument('-B', '--batch', type=int, default=1)
    parser.add_argument('model', type=str)
    return parser.parse_args(args)


def do_shape_infer(model, batch=1):
    for elem in model.graph.input:
        etype = elem.type
        if etype.HasField('tensor_type'):
            eshape = etype.tensor_type.shape
            for dim in eshape.dim:
                if dim.HasField('dim_param'):
                    dim.dim_value = batch
                break
    model = onnx.shape_inference.infer_shapes(model, True)
    return model


def do_split(options):
    model = onnx.load(options.model)
    model = do_shape_infer(model, options.batch)
    graph = model.graph

    name2input = {x.name: x for x in graph.input}
    name2output = {x.name: x for x in graph.output}
    name2initializer = {x.name: x for x in graph.initializer}
    name2vinfo = {x.name: x for x in graph.value_info}

    def node2model(node):
        inputs = []
        outputs = []
        initializer = []
        for name in node.input:
            if name in name2input:
                inputs.append(name2input[name])
            elif name in name2initializer:
                initializer.append(name2initializer[name])
            elif name in name2vinfo:
                inputs.append(name2vinfo[name])
            else:
                logger.warning(f'cannot find input {name}, convert node {node.name} failed')
                return None
        for name in node.output:
            if name in name2output:
                outputs.append(name2output[name])
            elif name in name2vinfo:
                outputs.append(name2vinfo[name])
            else:
                logger.warning(f'cannot find output {name}, convert node {node.name} failed')
                return None
        model = onnx.helper.make_model(
                    onnx.helper.make_graph(
                        [node],
                        node.name,
                        inputs,
                        outputs,
                        initializer
                    )
                )
        return model

    for node in graph.node:
        m = node2model(node)
        if not m:
            continue
        mpath = os.path.join(options.output, m.graph.name + '.onnx')
        with open(mpath, 'wb') as f:
            f.write(m.SerializeToString())


def main():
    options = parse_args()
    logging.basicConfig(level=logging.INFO)

    if options.action == 'split':
        do_split(options)


if __name__ == '__main__':
    main()
