#!/usr/bin/env python3
import argparse

import torch
import torch.nn as nn

from torch.ao.quantization.qconfig_mapping import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx

import torchvision.models as models


from imagenet_pipeline import ImageNetPipeline, transform_iterator


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path', type=str)
    parser.add_argument('-o', '--output')
    return parser.parse_args()


def do_prepare_fx(model_fp32, qconfig_mapping, example_inputs, is_qat=False):
    # TODO: figure out
    from torch.ao.quantization.fx.custom_config import PrepareCustomConfig
    from torch.ao.quantization.qconfig_mapping import QConfigMapping

    from torch.ao.quantization.fx.custom_config import FuseCustomConfig
    from torch.ao.quantization.fx.fuse import fuse
    from torch.ao.quantization.fx.prepare import prepare
    from torch.ao.quantization.fx.tracer import QuantizationTracer
    from torch.fx.graph_module import GraphModule

    prepare_custom_config = PrepareCustomConfig()
    equalization_config = QConfigMapping()

    tracer = QuantizationTracer([], [])
    graph_module = GraphModule(model_fp32, tracer.trace(model_fp32))
    for node in graph_module.graph.nodes:
        if not hasattr(node, "meta"):
            node.meta = {}

    fuse_custom_config = FuseCustomConfig()
    graph_module = fuse(graph_module, is_qat,
                        fuse_custom_config, None)

    prepared = prepare(
        graph_module,
        qconfig_mapping,
        is_qat,
        tracer.node_name_to_scope,
        example_inputs=example_inputs,
        prepare_custom_config=prepare_custom_config,
        _equalization_config=equalization_config,
        backend_config=None,
        is_standalone_module=False,
    )
    return prepared


def do_ptq(model_fp32, dataloader, example_inputs):
    qconfig_mapping = get_default_qconfig_mapping()
    model_prepared = do_prepare_fx(model_fp32, qconfig_mapping, example_inputs)
    # model_prepared = prepare_fx(model_fp32, qconfig_mapping, example_inputs)

    ImageNetPipeline.calibrate(model_prepared, dataloader)
    model_converted = convert_fx(model_prepared)
    return model_converted


def do_qat(model_fp32, dataloader, example_inputs):
    qconfig_mapping = get_default_qat_qconfig_mapping()
    model_prepared = do_prepare_fx(model_fp32, qconfig_mapping, example_inputs, True)
    # model_prepared = prepare_qat_fx(
    #     model_fp32, qconfig_mapping, example_inputs)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model_prepared.parameters(), lr=1e-3)
    ImageNetPipeline.train(model_prepared, dataloader, optimizer, loss_fn)
    model_converted = convert_fx(model_prepared)
    return model_converted


def main():
    options = parse_options()

    device = torch.device('cpu')

    example_inputs = tuple(x.to(device)
                           for x in ImageNetPipeline.get_example_inputs())
    dataloader = transform_iterator(ImageNetPipeline.get_dataloader(
        options.dataset_path), lambda t: (x.to(device) for x in t))

    model_fp32 = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT).eval().to(device)
    acc = ImageNetPipeline.eval(model_fp32, dataloader)
    print(f'origin model, acc: {acc * 100:.4f}%')

    # model_converted = do_ptq(model_fp32, dataloader, example_inputs)
    model_converted = do_qat(model_fp32, dataloader, example_inputs)
    acc = ImageNetPipeline.eval(model_converted, dataloader)
    print(f'converted model, acc: {acc * 100:.4f}%')

    if options.output:
        torch.onnx.export(
            model_converted,
            example_inputs,
            options.output,
        )


if __name__ == "__main__":
    main()
