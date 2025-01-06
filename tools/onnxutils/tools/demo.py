#!/usr/bin/env python3
import argparse


import torch
from torch import nn

from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver
from torch.ao.quantization.fake_quantize import FakeQuantize

from onnxutils.quantization.utils import symbolic_trace
from onnxutils.quantization.quantizer import BasicQuantizer

from unimodel_pipeline import ImageNetPipeline


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path')
    parser.add_argument('-o', '--output')
    return parser.parse_args()


def main():
    options = parse_options()

    example_inputs = (torch.rand(1, 3, 224, 224).cuda(),)
    dataloader = ImageNetPipeline.get_dataloader(options.dataset_path)

    graph_module = torch.hub.load(
        "pytorch/vision",
        "resnet50",
        weights="ResNet50_Weights.IMAGENET1K_V1"
    )
    graph_module.cuda()

    ImageNetPipeline.eval(graph_module, dataloader, 'cuda')

    graph_module = symbolic_trace(graph_module)
    quantizer = BasicQuantizer()
    graph_module = quantizer.quantize_modules(graph_module, [
        {
            'module_type': nn.Conv2d,
            'weight': FakeQuantize.with_args(
                observer=PerChannelMinMaxObserver
            ),
            'activation': FakeQuantize.with_args(observer=HistogramObserver)
        }
    ])
    ImageNetPipeline.train(graph_module, dataloader, 'cuda')
    ImageNetPipeline.eval(graph_module, dataloader, 'cuda')

    graph_module = quantizer.finalize(graph_module)

    if options.output:
        torch.onnx.export(
            graph_module,
            example_inputs,
            options.output,
            input_names=['x'],
            output_names=['y']
        )


if __name__ == "__main__":
    main()
