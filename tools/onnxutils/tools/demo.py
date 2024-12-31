#!/usr/bin/env python3
import argparse

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from onnxutils.common import OnnxModel
from onnxutils.optim import find_optimizer
from onnxutils.onnx2torch import convert


class RandomDataset:
    str2dtype = {
        'tensor(float)': np.float32
    }

    def __init__(self, fields, size):
        self.fields = fields
        self.size = size

        self.rewind()

    def rewind(self):
        self.idx = 0

    def get_next(self):
        if self.idx >= self.size:
            return None
        self.idx += 1
        return {
            f.name: np.random.rand(
                *f.shape).astype(RandomDataset.str2dtype[f.type])
            for f in self.fields
        }

    def __len__(self):
        return self.size

    def __next__(self):
        tmp = self.get_next()
        if tmp is not None:
            return tmp
        raise StopIteration

    def __iter__(self):
        self.rewind()
        return self


def optim_onnx(onnx_model: OnnxModel):
    optimizers = [
        'convert-constant-to-initializer',
        'convert-shape-to-initializer',
        'onnx-simplifier',
        'convert-shape-to-initializer',
    ]
    for x in optimizers:
        optimizer = find_optimizer(x)
        if optimizer is None:
            raise RuntimeError(f"can not find '{x}' optimizer, ignore")
        onnx_model = optimizer.apply(onnx_model)

    with onnx_model.session() as sess:
        for node in onnx_model.proto().graph.node:
            if node.name == '':
                node.name = sess.unique_name()
    return onnx_model


def verify_outputs(outputs0, outputs1, rtol=1e-2, atol=1e-3):
    assert outputs0.keys() == outputs1.keys()

    for k in outputs0.keys():
        val0 = outputs0[k]
        if isinstance(val0, torch.Tensor):
            val0 = val0.detach().cpu().numpy()

        val1 = outputs1[k]
        if isinstance(val1, torch.Tensor):
            val1 = val1.detach().cpu().numpy()

        is_ok = np.allclose(val0, val1, rtol, atol)
        if is_ok:
            print(f"verify field[{k}]...passed")
        else:
            print(f"verify field[{k}]...failed")


class LayerObserver(nn.Module):
    def __init__(self, t):
        super().__init__()
        self.target = t

    def forward(self, *args, **kwargs):
        return self.target(*args, **kwargs)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    onnx_model = optim_onnx(onnx_model)

    torch_module = convert(onnx_model)
    onnx_mapping = torch_module.onnx_mapping

    sess = ort.InferenceSession(
        options.model,
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])

    for name, m in torch_module.named_children():
        if isinstance(m, nn.Conv2d):
            setattr(torch_module, name, LayerObserver(m))

    dataset = RandomDataset(sess.get_inputs(), 10)
    for data in dataset:
        sess_outputs = sess.run(None, data)
        sess_outputs = {
            desc.name: val
            for val, desc in zip(sess_outputs, sess.get_outputs())
        }
        torch_outputs = torch_module(
            *tuple(torch.from_numpy(data[x]) for x in onnx_mapping.inputs))
        torch_outputs = {
            desc: val for val, desc in zip(torch_outputs, onnx_mapping.outputs)
        }
        verify_outputs(sess_outputs, torch_outputs)


if __name__ == "__main__":
    main()
