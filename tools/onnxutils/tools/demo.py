#!/usr/bin/env python3
import os
import argparse

import numpy as np
import torch

from onnxutils.common import OnnxModel, DatasetTransformer
from onnxutils.onnx2torch import convert


class UnimodelDataset:
    fields = [
        ("imgs", [10, 3, 576, 960], np.float32),
        ("fused_projection", [1, 10, 4, 4], np.float32),
        ("pose", [1, 4, 4], np.float32),
        ("prev_pose_inv", [1, 4, 4], np.float32),
        ("extrinsics", [1, 10, 4, 4], np.float32),
        ("norm_intrinsics", [1, 10, 4, 4], np.float32),
        ("distortion_coeff", [1, 10, 6], np.float32),
        ("prev_bev_feats", [1, 48, 60, 77], np.float32),
        ("sdmap_encode", [1, 9, 128, 160], np.float32),
        ("mpp", [1, 3, 224, 384], np.float32),
        ("mpp_pose_state", [1, 6], np.float32),
        ("mpp_valid", [1, 1], np.float32),
        ("prev_feat_stride16", [1, 48, 36, 60], np.float32),
        ("sdmap_mat", [1, 4, 240, 400], np.float32),
    ]

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.snippets = os.listdir(self.root_dir)

    def load_item(self, path):
        return {
            f[0]: torch.from_numpy(np.fromfile(os.path.join(
                path, f"{f[0]}.bin"), dtype=f[2]).reshape(f[1]))
            for f in UnimodelDataset.fields
        }

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        return self.load_item(os.path.join(self.root_dir, str(idx)))


def quantize_model(model, dataset, is_qat=False):
    assert not is_qat
    return model


# 'convert-constant-to-initializer',
# 'convert-shape-to-initializer',
# 'onnx-simplifier',
# 'convert-shape-to-initializer',
def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path')
    parser.add_argument('-o', '--output')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    onnx_model = OnnxModel.from_file(options.model)
    with onnx_model.session() as sess:
        for node in onnx_model.proto().graph.node:
            if node.name == '':
                node.name = sess.unique_name()

    torch_model = convert(onnx_model).cuda()
    onnx_mapping = torch_model.onnx_mapping

    dataset = UnimodelDataset(options.dataset_path)
    dataset = DatasetTransformer(dataset,
                                 lambda item: tuple(
                                     item[x].cuda()
                                     for x in onnx_mapping.inputs
                                 ))

    model_quantized = quantize_model(torch_model, dataset)

    if options.output:
        torch.onnx.export(
            model_quantized,
            dataset[0],
            options.output,
            input_names=onnx_mapping.inputs
        )


if __name__ == "__main__":
    main()
