#!/usr/bin/env python3

import torch


class MyModule(torch.nn.Module):
    def __init__(self, N, M):
        super(MyModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(N, M))

    def forward(self, input):
        if input.sum() > 0:
            output = self.weight.mv(input)
        else:
            output = self.weight + input
        return output


my_module = MyModule(4, 4)
sm = torch.jit.script(my_module)

sm.save("traced_model.pt")
