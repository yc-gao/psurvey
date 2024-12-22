#!/usr/bin/env python3
import argparse
import warnings

from onnxoptim import OnnxModel, find_optimizer


def verify_model(origin_model, new_model, rtol=1e-4, atol=1e-5):
    import numpy as np
    import onnxruntime as ort

    def random_tensor(node):
        str2dtype = {
            'tensor(float)': np.float32
        }
        return np.random.rand(*node.shape).astype(str2dtype[node.type])

    sess0 = ort.InferenceSession(
        origin_model.model().SerializeToString(),
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])
    sess1 = ort.InferenceSession(
        new_model.model().SerializeToString(),
        providers=[
            x for x in ['CUDAExecutionProvider', 'CPUExecutionProvider']
            if x in ort.get_available_providers()
        ])

    ret = True
    example_inputs = {x.name: random_tensor(x) for x in sess0.get_inputs()}
    sess0_outputs = sess0.run(None, example_inputs)
    sess1_outputs = sess1.run(None, example_inputs)

    ret = [
        (node.name, np.allclose(output0, output1, rtol=rtol, atol=atol))
        for node, output0, output1 in zip(sess0.get_outputs(), sess0_outputs, sess1_outputs)]
    for name, ok in ret:
        print(f"verify {name}...", ok)
    return all(ret)


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output')
    parser.add_argument('--optimizer', action='append', default=[])
    parser.add_argument('--verify', type=int, default=10)
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    origin_model = OnnxModel(options.model)

    onnx_model = origin_model.clone()
    for x in options.optimizer:
        optimizer = find_optimizer(x)
        if optimizer is None:
            warnings.warn(f"can not find '{x}' optimizer, ignore")
            continue
        onnx_model = optimizer.apply(onnx_model)

    if any(not verify_model(
            origin_model, onnx_model) for _ in range(options.verify)):
        warnings.warn("verify model failed")

    if options.output:
        onnx_model.save(options.output)


if __name__ == "__main__":
    main()
