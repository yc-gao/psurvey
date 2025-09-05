#!/usr/bin/env python3
import argparse

import torch

import torch.nn as nn
import torch.nn.functional as F

from torchvision import models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    model = models.resnet50(models.ResNet50_Weights.IMAGENET1K_V2).to('cuda')
    torch_input = (torch.ones(1, 3, 224, 224).to('cuda'), )
    if args.output:
        torch.onnx.export(
            model,
            torch_input,
            args.output,
            input_names=["x"],
            output_names=["y"]
        )
    else:
        model.eval()
        y = model(*torch_input)
        print(y)


if __name__ == '__main__':
    main()
