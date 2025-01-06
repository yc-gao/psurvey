import torch
from torch import nn
import torch.nn.functional as F

from .common import BasicQuantizedModule


class QuantizedConvTransposeNd(nn.modules.conv._ConvTransposeNd, BasicQuantizedModule):
    @staticmethod
    def from_float(cls, float_conv, weight_qparams):
        q_conv = cls(
            float_conv.in_channels,
            float_conv.out_channels,
            float_conv.kernel_size,  # type: ignore[arg-type]
            float_conv.stride,  # type: ignore[arg-type]
            float_conv.padding,  # type: ignore[arg-type]
            float_conv.output_padding,  # type: ignore[arg-type]
            float_conv.groups,
            float_conv.bias is not None,  # type: ignore[arg-type]
            float_conv.dilation,  # type: ignore[arg-type]
            float_conv.padding_mode,
            device=float_conv.weight.device,
            dtype=float_conv.weight.dtype,
            weight_qparams=weight_qparams,
        )
        q_conv.weight = torch.nn.Parameter(float_conv.weight.detach())
        if float_conv.bias is not None:
            q_conv.bias = torch.nn.Parameter(float_conv.bias.detach())
        return q_conv


class QuantizedConvTranspose1d(nn.ConvTranspose1d, QuantizedConvTransposeNd):
    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        return QuantizedConvTransposeNd.from_float(cls, float_conv, weight_qparams)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict = {}
    ):
        nn.ConvTranspose1d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(
            x,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            1,
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose1d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result


class QuantizedConvTranspose2d(nn.ConvTranspose2d, QuantizedConvTransposeNd):
    @classmethod
    def from_float(cls, float_conv, weight_qparams):
        return QuantizedConvTransposeNd.from_float(cls, float_conv, weight_qparams)

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        output_padding=0,
        groups=1,
        bias=True,
        dilation=1,
        padding_mode="zeros",
        device=None,
        dtype=None,
        weight_qparams: dict = {}
    ):
        nn.ConvTranspose2d.__init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
            device,
            dtype,
        )
        self._init_weight_qparams(weight_qparams, device)

    def forward(self, x, output_size=None):
        output_padding = self._output_padding(
            x,  # type: ignore[arg-type]
            output_size,
            self.stride,  # type: ignore[arg-type]
            self.padding,  # type: ignore[arg-type]
            self.kernel_size,  # type: ignore[arg-type]
            2,
            self.dilation,  # type: ignore[arg-type]
        )

        weight_quant_dequant = self.get_weight()
        result = F.conv_transpose2d(
            x,
            weight_quant_dequant,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
        return result
