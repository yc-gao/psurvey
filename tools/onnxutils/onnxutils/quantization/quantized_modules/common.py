import torch
import torch.nn as nn


class BasicQuantizedModule(nn.Module):
    def _init_weight_qparams(self, weight_qparams):
        self.weight_qscheme = weight_qparams['qscheme']

        self.weight_scale = weight_qparams['scale']
        self.weight_zero_point = weight_qparams['zero_point']

        self.weight_quant_min = weight_qparams['quant_min']
        self.weight_quant_max = weight_qparams['quant_max']

        assert self.weight_qscheme in (torch.per_channel_affine,
                                       torch.per_tensor_affine,)

        if self.weight_qscheme == torch.per_channel_affine:
            self.weight_axis_int = weight_qparams['ch_axis']

    def get_fake_quant_weight(self):
        if self.weight_qscheme == torch.per_channel_affine:
            return torch.fake_quantize_per_channel_affine(
                self.weight,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_quant_min,
                self.weight_quant_max
            )
        return torch.fake_quantize_per_tensor_affine(
            self.weight,
            self.weight_scale,
            self.weight_zero_point,
            self.weight_quant_min,
            self.weight_quant_max
        )
