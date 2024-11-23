#!/usr/bin/env python3
import argparse
import random
import tempfile
from pathlib import Path

import numpy as np

import onnx

from onnxruntime.quantization.quant_utils import QuantType, QuantizationMode
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, create_calibrator
from onnxruntime.quantization.qdq_quantizer import QDQQuantizer
from onnxruntime.quantization.onnx_quantizer import ONNXQuantizer
from onnxruntime.quantization.registry import QLinearOpsRegistry, QDQRegistry


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
    parser.add_argument('--cache')
    parser.add_argument('-o', '--output', type=str, default='output.onnx')
    parser.add_argument('-f', '--format', type=str,
                        default='qdq', choices=['qdq', 'qop'])
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()
    calibration_data_reader = FakeResnetCalibrationDataReader(16)

    extra_options = {}
    op_types_to_quantize = []
    nodes_to_quantize = []
    nodes_to_exclude = []

    if not op_types_to_quantize:
        q_linear_ops = list(QLinearOpsRegistry.keys())
        qdq_ops = list(QDQRegistry.keys())
        op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    with tempfile.TemporaryDirectory() as tmp:
        if options.cache:
            Path(options.cache).mkdir(parents=True, exist_ok=True)
            tmp = options.cache
        model_path = Path(options.model)
        inferred_model_path = Path(
            options.cache or tmp)/(model_path.stem + "-inferred" + model_path.suffix)
        onnx.shape_inference.infer_shapes_path(
            str(model_path), str(inferred_model_path))
        model = onnx.load(inferred_model_path)

        calibrator = create_calibrator(
            inferred_model_path,
            op_types_to_quantize,
            augmented_model_path=Path(inferred_model_path).parent.joinpath(
                "augmented_model.onnx").as_posix(),
            calibrate_method=CalibrationMethod.MinMax,
        )
        calibrator.collect_data(calibration_data_reader)
        tensors_range = calibrator.compute_data()
        del calibrator

        if options.format == 'qop':
            quantizer = ONNXQuantizer(
                model,
                False,
                False,
                QuantizationMode.QLinearOps,
                True,
                QuantType.QInt8,
                QuantType.QInt8,
                tensors_range,
                nodes_to_quantize,
                nodes_to_exclude,
                op_types_to_quantize,
                extra_options
            )
        else:
            quantizer = QDQQuantizer(
                model,
                False,
                False,
                QuantType.QInt8,
                QuantType.QInt8,
                tensors_range,
                nodes_to_quantize,
                nodes_to_exclude,
                op_types_to_quantize,
                extra_options
            )
        quantizer.quantize_model()
        if options.output:
            quantizer.model.save_model_to_file(options.output)


if __name__ == "__main__":
    main()
