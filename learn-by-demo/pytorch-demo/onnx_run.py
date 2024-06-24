#!/usr/bin/env python3
import onnx
import torch
import onnxruntime as ort
import numpy as np


def main():
    onnx_model = onnx.load("output.onnx")
    onnx.checker.check_model(onnx_model)

    sess_options = ort.SessionOptions()
    sess_options.enable_profiling = True
    sess_options.log_severity_level = 0
    sess_options.optimized_model_filepath = "optimized_model.onnx"

    providers = [
        ("CUDAExecutionProvider", {"device_id": torch.cuda.current_device(),
                                   "user_compute_stream": str(torch.cuda.current_stream().cuda_stream)})]

    ort_sess = ort.InferenceSession(
        'output.onnx', sess_options, providers=providers)
    y = ort_sess.run(None, {'x': np.random.rand(
        1, 1, 32, 32).astype(np.float32)})
    print(y)


if __name__ == '__main__':
    main()
