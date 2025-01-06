#!/usr/bin/env python3
import argparse

import torch
from torch import nn

from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.fx.tracer import QuantizationTracer

from onnxutils.quantization.utils import symbolic_trace
from onnxutils.quantization.convert_observer_or_fq import ConvertObserverOrFq


class M(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(3, 3, 3, padding=1)
        self.conv1 = nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        return x


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    return parser.parse_args()


def main():
    options = parse_options()

    example_inputs = (torch.rand(1, 3, 224, 224),)
    module = M()

    graph_module = symbolic_trace(module)
    graph_module.print_readable()

    for node in graph_module.graph.nodes:
        if node.target == 'conv0':
            graph_module.add_submodule(
                'fq0', FakeQuantize(observer=MinMaxObserver))
            with graph_module.graph.inserting_after(node):
                new_node = graph_module.graph.create_node('call_module', 'fq0')
                node.replace_all_uses_with(new_node)
                new_node.insert_arg(0, node)
    graph_module = torch.fx.GraphModule(graph_module, graph_module.graph)
    graph_module.print_readable()

    graph_module(*example_inputs)

    graph_module = ConvertObserverOrFq.apply(graph_module)
    graph_module.print_readable()

    # graph_module = ConvertObserverOrFq.apply(graph_module)
    # graph_module.print_readable()

    if options.output:
        torch.onnx.export(
            graph_module,
            example_inputs,
            options.output,
        )


if __name__ == "__main__":
    main()
