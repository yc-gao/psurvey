import torch
from torch import nn
import torch.nn.functional as F

from .common import BasicQuantizedModule


class QuantizedConvNd(nn.modules.conv._ConvNd, BasicQuantizedModule):
    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        q_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        q_conv.weight = nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            q_conv.bias = nn.Parameter(float_conv.bias.detach())
        return q_conv


class QuantizedConv1d(QuantizedConvNd, nn.Conv1d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
        weight_qparams: dict = {}
    ):
        nn.Conv1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams)

    def forward(self, x):
        weight_quant_dequant = self.get_weight()
        result = F.conv1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result


class QuantizedConv2d(QuantizedConvNd, torch.nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict = {},
    ):
        torch.nn.Conv2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams)

    def forward(self, x):
        weight_quant_dequant = self.get_weight()

        result = F.conv2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result
