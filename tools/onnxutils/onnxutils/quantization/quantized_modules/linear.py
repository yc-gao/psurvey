import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BasicQuantizedModule


class QuantizedLinear(BasicQuantizedModule, nn.Linear):
    @classmethod
    def from_qat(cls, qat_module):
        fq = qat_module.weight_fake_quant
        qscheme = fq.qscheme
        if qscheme == torch.per_tensor_symmetric:
            qscheme = torch.per_tensor_affine
        if qscheme == torch.per_channel_symmetric:
            qscheme = torch.per_channel_affine

        scale, zero_point = fq.calculate_qparams()
        weight_qparams = {
            'qscheme': qscheme,
            'quant_min': fq.quant_min,
            'quant_max': fq.quant_max,
            'scale': scale,
            'zero_point': zero_point,
        }
        if qscheme == torch.per_channel_affine:
            weight_qparams['ch_axis'] = fq.ch_axis

        q_linear = cls(
            qat_module.in_features,
            qat_module.out_features,
            qat_module.bias is not None,
            device=qat_module.weight.device,
            dtype=qat_module.weight.dtype,
            weight_qparams=weight_qparams,
        )
        q_linear.weight = nn.Parameter(qat_module.weight.detach())
        if qat_module.bias is not None:
            q_linear.bias = nn.Parameter(qat_module.bias.detach())
        return q_linear

    def __init__(
            self,
            in_features,
            out_features,
            bias=True,
            device=None,
            dtype=None,
            weight_qparams: dict = {}) -> None:
        nn.Linear.__init__(self, in_features, out_features,
                           bias, device, dtype)
        self._init_weight_qparams(weight_qparams)

    def forward(self, x):
        weight = self.get_fake_quant_weight()
        return F.linear(x, weight, self.bias)
