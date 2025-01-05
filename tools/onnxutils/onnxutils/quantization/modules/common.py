import torch
from torch import nn


class BasicQuantizedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def _init_weight_qparams(self, weight_qparams):
        self.weight_qscheme = weight_qparams['qscheme']
        self.weight_dtype = weight_qparams['dtype']
        self.weight_scale = weight_qparams['scale']
        self.weight_zero_point = weight_qparams['zero_point']

        assert self.weight_qscheme in (torch.per_channel_affine,
                                       torch.per_tensor_affine,)

        if self.weight_qscheme == torch.per_channel_affine:
            self.weight_axis_int = weight_qparams['axis']

    def get_weight(self):
        if self.weight_qscheme == torch.per_channel_affine:
            weight = torch.quantize_per_channel(
                self.weight,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_axis_int,
                self.weight_dtype
            )
        else:
            weight = torch.quantize_per_tensor(
                self.weight,
                self.weight_scale,
                self.weight_zero_point,
                self.weight_dtype
            )
        weight = weight.dequantize()
        return weight
