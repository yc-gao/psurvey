#!/usr/bin/env python3
import os
import argparse

import numpy as np
import onnxruntime as ort


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
            f[0]: np.fromfile(os.path.join(
                path, f"{f[0]}.bin"), dtype=f[2]).reshape(f[1])
            for f in UnimodelDataset.fields
        }

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        return self.load_item(os.path.join(self.root_dir, str(idx)))


def do_verify(model0, model1, dataset, rtol=1e-2, atol=1e-3):
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

    for idx in range(len(dataset)):
        data = dataset[idx]
        outputs0 = sess0.run(None, data)
        outputs1 = sess1.run(None, data)

        outputs0 = {node.name: output for node,
                    output in zip(sess0.get_outputs(), outputs0)}
        outputs1 = {node.name: output for node,
                    output in zip(sess1.get_outputs(), outputs1)}

        for name in outputs0.keys():
            output0 = outputs0[name]
            output1 = outputs1[name]

            is_ok = np.allclose(output0, output1, rtol, atol)
            if not is_ok:
                print(f"verify field[{name}] failed")
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
                print(f"diff {max_val} {min_val} {mean_val} {tmp.argmax()}")
                if name == 'refline_instance_confidence':
                    print(output0.reshape(-1))
                    print(output1.reshape(-1))
            else:
                print(f"verify field[{name}]...passed")


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-path')
    parser.add_argument('model0')
    parser.add_argument('model1')
    return parser.parse_args()


def main():
    options = parse_options()
    dataset = UnimodelDataset(options.dataset_path)
    do_verify(options.model0, options.model1, dataset)


if __name__ == "__main__":
    main()
