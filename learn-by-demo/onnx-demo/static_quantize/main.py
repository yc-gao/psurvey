#!/usr/bin/env python3
import argparse
import random

import numpy as np

from onnxruntime.quantization import QuantFormat, QuantType, quantize_static
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod


class FakeResnetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, batch_size: int = 16):
        super().__init__()
        self.dataset = [
            (np.random.rand(1, 3, 224, 224).astype(np.float32), random.randint(0, 999)) for _ in range(batch_size)
        ]
        self.iterator = iter(self.dataset)

    def get_next(self) -> dict:
        try:
            return {"data": next(self.iterator)[0]}
        except Exception:
            return None


# https://github.com/onnx/models/blob/main/validated/vision/classification/resnet/model/resnet50-v2-7.onnx
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output.onnx')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    dataloader = FakeResnetCalibrationDataReader(16)
    quantize_static(
        options.model,
        options.output,
        calibration_data_reader=dataloader)


if __name__ == "__main__":
    main()
