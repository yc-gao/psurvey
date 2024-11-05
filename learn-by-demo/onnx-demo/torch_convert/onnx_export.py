#!/usr/bin/env python3
import argparse

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output.onnx')
    args = parser.parse_args()

    torch_model = NeuralNetwork()
    torch_input = torch.randn(1, 1, 28, 28)
    torch.onnx.export(
        torch_model, (torch_input, ), args.output, input_names=['x'])


if __name__ == '__main__':
    main()
