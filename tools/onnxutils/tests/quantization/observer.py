#!/usr/bin/env python3
import unittest

import torch

from onnxutils.onnx2torch.utils import OnnxMapping
from onnxutils.quantization.layer_observer import LayerObserver


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv1 = torch.nn.Conv2d(3, 3, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(3, 3, 3, padding=1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ObserverTests(unittest.TestCase):
    def test_conv0(self):
        model = M()
        model.conv2.onnx_mapping = OnnxMapping(
            inputs=tuple(),
            outputs=('conv2',))
        with LayerObserver.observe(model, 'conv2') as observer:
            for _ in range(10):
                example_inputs = (torch.rand(1, 3, 224, 224), )
                val = model(*example_inputs)
                self.assertTrue(torch.allclose(val, observer.value('conv2')))


if __name__ == '__main__':
    unittest.main()
