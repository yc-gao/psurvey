#!/usr/bin/env python3
import argparse
from collections import OrderedDict

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from onnxutils.common import OnnxModel
from onnxutils.onnx2torch import convert


class RandomDataset:
    str2dtype = {
        'tensor(float)': np.float32
    }

    def __init__(self, fields, size):
        self.fields = fields
        self.size = size

        self.rewind()

    def load_item(self, _):
        return {
            f.name: np.random.rand(
                *f.shape).astype(RandomDataset.str2dtype[f.type])
            for f in self.fields
        }

    def rewind(self):
        self.idx = 0

    def get_next(self):
        if self.idx >= self.size:
            return None
        self.idx += 1
        return self.load_item(self.idx - 1)

    def __getitem__(self, idx):
        return self.load_item(idx)

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


class LayerObserver(nn.Module):
    def __init__(self, collector, t):
        super().__init__()
        self.collector = collector
        self.target = t

        if hasattr(t, 'onnx_mapping'):
            self.onnx_mapping = t.onnx_mapping

    def record(self, vals):
        if not isinstance(vals, (tuple, list)):
            vals = (vals, )
        for name, val in zip(self.onnx_mapping.outputs, vals):
            self.collector[name] = val

    def forward(self, *args, **kwargs):
        t = self.target(*args, **kwargs)
        self.record(t)
        return t


# 'convert-constant-to-initializer',
# 'convert-shape-to-initializer',
# 'onnx-simplifier',
# 'convert-shape-to-initializer',
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    with onnx_model.session() as sess:
        for node in onnx_model.proto().graph.node:
            if node.name == '':
                node.name = sess.unique_name()

    torch_module = convert(onnx_model).eval()
    onnx_mapping = torch_module.onnx_mapping

    data_collector = OrderedDict()
    for name, m in torch_module.named_children():
        if name == 'initializers':
            continue
        setattr(torch_module, name, LayerObserver(data_collector, m))

    sess = ort.InferenceSession(
        onnx_model.proto().SerializeToString(),
        providers=['CPUExecutionProvider'])

    dataset = RandomDataset(sess.get_inputs(), 10)
    example_inputs = dataset[0]

    torch_module(
        *tuple(torch.from_numpy(example_inputs[x]) for x in onnx_mapping.inputs))

    with onnx_model.session() as sess:
        for k in data_collector.keys():
            sess.add_output(onnx_model.get_vinfo_by_name(k))
    sess = ort.InferenceSession(
        onnx_model.proto().SerializeToString(),
        providers=['CPUExecutionProvider'])
    for data in dataset:
        torch_module(
            *tuple(torch.from_numpy(data[x]) for x in onnx_mapping.inputs))
        sess_outputs = {
            desc.name: val
            for val, desc in zip(sess.run(None, data), sess.get_outputs())
        }

        for k, torch_val in data_collector.items():
            if torch_val.dtype == torch.bool:
                continue
            onnx_val = torch.from_numpy(sess_outputs[k])
            is_ok = torch.allclose(
                torch_val,
                onnx_val,
                1e-3,
                max(1e-3, onnx_val.abs().max() * 1e-3)
            )
            print(f"verify {k}...", is_ok)
            if not is_ok:
                print(f"onnx {onnx_val.max()} {onnx_val.min()} {onnx_val.mean()}")
                tmp = (torch_val - onnx_val).abs()
                print(f"diff {tmp.max()} {tmp.min()} {tmp.mean()}")
        break


if __name__ == "__main__":
    main()
