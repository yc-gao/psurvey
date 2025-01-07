import os

import torch

import numpy as np


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

    def __init__(self, root_dir, input_names=None):
        self.root_dir = root_dir
        self.input_names = input_names

        self.snippets = os.listdir(self.root_dir)

    def load_item(self, path):
        vals = {
            f[0]: torch.from_numpy(np.fromfile(os.path.join(
                path, f"{f[0]}.bin"), dtype=f[2]).reshape(f[1]))
            for f in UnimodelDataset.fields
        }
        if self.input_names is None:
            return vals
        return tuple(vals[input_name] for input_name in self.input_names)

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, idx):
        return self.load_item(os.path.join(self.root_dir, str(idx)))
