import torch
from torch import nn
import torch.nn.functional as F

from .common import BasicQuantizedModule


class Linear(nn.Linear, BasicQuantizedModule):
    @classmethod
    def from_float(cls, float_linear, weight_qparams):
        q_linear = Linear(
            float_linear.in_features,
            float_linear.out_features,
            float_linear.bias is not None,
            device=float_linear.weight.device,
            dtype=float_linear.weight.dtype,
            weight_qparams=weight_qparams,
        )
        q_linear.weight = torch.nn.Parameter(float_linear.weight.detach())
        if float_linear.bias is not None:
            q_linear.bias = torch.nn.Parameter(float_linear.bias.detach())
        return q_linear

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        device=None,
        dtype=None,
        weight_qparams: dict = {}
    ):
        nn.Linear.__init__(in_features, out_features, bias, device, dtype)
        self._init_weight_qparams(weight_qparams)

    def forward(self, x):
        weight_quant_dequant = self.get_weight()
        result = F.linear(x, weight_quant_dequant, self.bias)
        return result
