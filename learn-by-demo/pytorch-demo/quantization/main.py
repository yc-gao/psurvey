#!/usr/bin/env python3
import argparse

import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx


from imagenet_pipeline import ImageNetPipeline, transform_iterator


class M(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 5, 3)
        self.bn0 = torch.nn.BatchNorm2d(5)
        self.relu0 = torch.nn.ReLU()

        self.conv1 = torch.nn.Conv2d(5, 5, 3)
        self.bn1 = torch.nn.BatchNorm2d(5)
        self.relu1 = torch.nn.ReLU()

        self.conv2 = torch.nn.Conv2d(5, 5, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        return x


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path', help="Imagenet eval image", type=str, required=True)
    parser.add_argument('-o', '--output')
    return parser.parse_args()


def main():
    options = parse_options()

    dataloader = ImageNetPipeline.get_dataloader(options.dataset_path)

    float_model = M().eval()

    backend = 'x86'
    qconfig_mapping = get_default_qconfig_mapping(backend)
    # qconfig_mapping = get_default_qconfig_mapping("fbgemm")
    # torch.backends.quantized.engine = backend

    example_inputs = (torch.randn(1, 3, 224, 224),)
    prepared_model = prepare_fx(float_model, qconfig_mapping, example_inputs)
    # print(prepared_model.code)

    ImageNetPipeline.calibrate(prepared_model, dataloader)
    quantized_model = convert_fx(prepared_model)
    # print(quantized_model.code)

    # acc, _, _ = ImageNetPipeline.eval(prepared_model, dataloader)
    # print(f"model, acc: {acc * 100:.4f}%")

    if options.output:
        torch.onnx.export(
            quantized_model,
            (next(iter(dataloader))[0], ),
            options.output,
        )


if __name__ == "__main__":
    main()
