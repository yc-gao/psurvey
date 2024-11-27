#!/usr/bin/env python3
import argparse

import torch

import torchvision

from imagenet_pipeline import ImageNetPipeline, transform_iterator


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path', help="Imagenet eval image", type=str, required=True)
    parser.add_argument('-o', '--output')
    return parser.parse_args()


def main():
    options = parse_options()

    device = torch.device('cpu')
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')

    dataloader = transform_iterator(
        ImageNetPipeline.get_dataloader(options.dataset_path), lambda items: [x.to(device) for x in items])

    model_quantized = torchvision.models.quantization.mobilenet_v2(
        pretrained=True, quantize=True)
    model_quantized = model_quantized.to(device)

    acc, _, _ = ImageNetPipeline.eval(model_quantized, dataloader)
    print(f"model, acc: {acc * 100:.4f}%")

    if options.output:
        torch.onnx.export(
            model_quantized,
            (next(iter(dataloader))[0], ),
            options.output,
        )


if __name__ == "__main__":
    main()
