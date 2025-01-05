import torch
from torch import nn
import torch.nn.functional as F


class QuantizedConvNd(nn.modules.conv._ConvNd):
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
