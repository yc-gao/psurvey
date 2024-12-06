#!/usr/bin/env python3

import torch
from torch import nn


class M(nn.Module):
    def __init__(self, n=10):
        super().__init__()
        self.n = n

    def forward(self, x):
        return x + 1 + self.n


def transform(m: nn.Module) -> torch.nn.Module:
    gm: torch.fx.GraphModule = torch.fx.symbolic_trace(m)
    for node in gm.graph.nodes:
        if node.op == 'call_function' and node.args[1] == 1:
            node.update_arg(1, 2)

    gm.graph.lint()
    gm.recompile()
    return gm


def main():
    module = M()
    module = transform(module)
    print(module)


if __name__ == "__main__":
    main()
