#!/usr/bin/env python3
import unittest

import os
import tempfile

import torch
import numpy as np
import onnx
import onnxruntime as ort

from onnxutils.onnx import OnnxModel
from onnxutils.optim import apply_optimizers
from onnxutils.onnx2torch import convert


class ConvTests(unittest.TestCase):
    def test_bn0(self):
        torch.set_printoptions(precision=8)
        torch_module = torch.nn.BatchNorm2d(64)
        with tempfile.TemporaryDirectory() as workdir:
            onnx_fpath = os.path.join(workdir, 'output.onnx')
            torch.onnx.export(
                torch_module,
                (torch.rand(3, 64, 144, 240),),
                onnx_fpath,
                input_names=['x'],
            )
            del torch_module

            sess = ort.InferenceSession(
                onnx_fpath,
                providers=['CPUExecutionProvider'])

            onnx.checker.check_model(onnx_fpath)
            onnx_model = OnnxModel.from_file(onnx_fpath)
            onnx_model = apply_optimizers(onnx_model, ['eliminate-identity'])
            torch_module = convert(onnx_model)
            torch_module.eval()
        for _ in range(100):
            x = np.random.rand(3, 64, 144, 240).astype(np.float32)
            y, = sess.run(None, {'x': x})
            pred = torch_module(torch.from_numpy(x))
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
