#!/usr/bin/env python3
import argparse

import torch
from torch import nn

from torch.ao.quantization.observer import MinMaxObserver, HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize


from onnxutils.quantization.utils import symbolic_trace
from onnxutils.quantization.qnn_quantizer import QnnQuantizer


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

    quantizer = QnnQuantizer()
    graph_module = quantizer.quantize(
        graph_module,
        [
            {
                'name': 'conv0',
                'weight': FakeQuantize.with_args(observer=PerChannelMinMaxObserver.with_args(qscheme=torch.per_channel_symmetric, dtype=torch.qint8)),
                'input': FakeQuantize.with_args(observer=HistogramObserver),
                'output': FakeQuantize.with_args(observer=HistogramObserver),

            },
        ]
    )
    graph_module.print_readable()

    graph_module = quantizer.finalize(graph_module)
    graph_module.print_readable()

    if options.output:
        torch.onnx.export(
            graph_module,
            example_inputs,
            options.output
        )


if __name__ == "__main__":
    main()
