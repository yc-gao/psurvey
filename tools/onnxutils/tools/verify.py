#!/usr/bin/env python3
import argparse

import numpy as np
import onnxruntime as ort


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


def do_verify(model0, model1, rtol=1e-4, atol=1e-4):
    sess0 = ort.InferenceSession(
        model0,
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])
    sess1 = ort.InferenceSession(
        model1,
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])
    dataset = RandomDataset(sess0.get_inputs(), 10)
    for data in dataset:
        sess0_outputs = sess0.run(None, data)
        sess1_outputs = sess1.run(None, data)
        for node, output0, output1 in zip(sess0.get_outputs(), sess0_outputs, sess1_outputs):
            is_ok = np.allclose(output0, output1, rtol, atol)
            if not is_ok:
                print(f"verify field[{node.name}] failed")
                max_val = output0.max()
                min_val = output0.min()
                mean_val = output0.mean()
                print(f"output0 {max_val} {min_val} {mean_val}")
                max_val = output1.max()
                min_val = output1.min()
                mean_val = output1.mean()
                print(f"output1 {max_val} {min_val} {mean_val}")
                tmp = np.absolute(output0 - output1)
                max_val = tmp.max()
                min_val = tmp.min()
                mean_val = tmp.mean()
                print(f"diff {max_val} {min_val} {mean_val}")
            else:
                print(f"verify field[{node.name}]...passed")


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('model0')
    parser.add_argument('model1')
    return parser.parse_args()


def main():
    options = parse_options()
    do_verify(options.model0, options.model1)


if __name__ == "__main__":
    main()
