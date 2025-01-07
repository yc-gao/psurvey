#!/usr/bin/env python3
import unittest

import os
import tempfile

import torch
import numpy as np
import onnx
import onnxruntime as ort

from onnxutils.onnx import OnnxModel
from onnxutils.onnx2torch import convert


class ConvTests(unittest.TestCase):
    def test_conv0(self):
        torch.set_printoptions(precision=8)
        torch_module = torch.nn.Conv2d(
            128,
            128,
            kernel_size=(1, 3),
            stride=(1, 1),
            padding=(0, 1),
            dilation=(1, 1),
            groups=1,
        )
        with tempfile.TemporaryDirectory() as workdir:
            onnx_fpath = os.path.join(workdir, 'output.onnx')
            torch.onnx.export(
                torch_module,
                (torch.rand(10, 128, 72, 120),),
                onnx_fpath,
                input_names=['x'],
            )
            del torch_module

            onnx.checker.check_model(onnx_fpath)
            onnx_model = OnnxModel.from_file(onnx_fpath)
            onnx_module = convert(onnx_model)
            sess = ort.InferenceSession(
                onnx_fpath,
                providers=['CPUExecutionProvider'])
        for _ in range(100):
            x = np.random.rand(10, 128, 72, 120).astype(np.float32)
            y, = sess.run(None, {'x': x})
            pred = onnx_module(torch.from_numpy(x))
            self.assertTrue(
                np.allclose(
                    y,
                    pred.detach().cpu().numpy(),
                    1e-5,
                    1e-5
                )
            )


if __name__ == '__main__':
    unittest.main()
