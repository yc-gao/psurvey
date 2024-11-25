#!/usr/bin/env python3
import argparse

import torch

from torchvision import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output.onnx')
    args = parser.parse_args()

    model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2)
    torch_input = (torch.randn(1, 3, 224, 224), )
    torch.onnx.export(
        model, torch_input, args.output)


if __name__ == '__main__':
    main()