#!/usr/bin/env python3
import argparse

import onnx
from onnx import numpy_helper
import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def onnx2pmodel(onnx_model):
    torch_model = NeuralNetwork()

    initalizers = dict()
    for init in onnx_model.graph.initializer:
        initalizers[init.name] = numpy_helper.to_array(init)
    for name, p in torch_model.named_parameters():
        p.data = torch.from_numpy(initalizers[name].copy()).data
    return torch_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx', type=str, default='output.onnx')
    args = parser.parse_args()

    onnx_model = onnx.load(args.onnx)
    pytorch_model = onnx2pmodel(onnx_model)
    print(pytorch_model)


if __name__ == '__main__':
    main()
