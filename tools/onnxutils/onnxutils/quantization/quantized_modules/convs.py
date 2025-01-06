import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import BasicQuantizedModule


class QuantizedConvNd(BasicQuantizedModule):
    @staticmethod
    def _from_qat(cls, qat_module):
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

        q_conv = cls(
            qat_module.in_channels,
            qat_module.out_channels,
            qat_module.kernel_size,  # type: ignore[arg-type]
            qat_module.stride,  # type: ignore[arg-type]
            qat_module.padding,  # type: ignore[arg-type]
            qat_module.dilation,  # type: ignore[arg-type]
            qat_module.groups,
            qat_module.bias is not None,  # type: ignore[arg-type]
            qat_module.padding_mode,
            device=qat_module.weight.device,
            dtype=qat_module.weight.dtype,
            weight_qparams=weight_qparams,
        )
        q_conv.weight = nn.Parameter(qat_module.weight.detach())
        if qat_module.bias is not None:
            q_conv.bias = nn.Parameter(qat_module.bias.detach())
        return q_conv


class QuantizedConv1d(QuantizedConvNd, nn.Conv1d):
    @classmethod
    def from_qat(cls, qat_module):
        return QuantizedConvNd._from_qat(cls, qat_module)

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
    ) -> None:

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
            dtype
        )
        self._init_weight_qparams(weight_qparams)

    def forward(self, x):
        assert self.padding_mode == 'zeros'

        weight = self.get_fake_quant_weight()
        result = F.conv1d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result


class QuantizedConv2d(BasicQuantizedModule, nn.Conv2d):
    @classmethod
    def from_qat(cls, qat_module):
        return QuantizedConvNd._from_qat(cls, qat_module)

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
            weight_qparams: dict = {},) -> None:

        nn.Conv2d.__init__(
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
            dtype
        )
        self._init_weight_qparams(weight_qparams)

    def forward(self, x):
        assert self.padding_mode == 'zeros'

        weight = self.get_fake_quant_weight()
        result = F.conv2d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result


class QuantizedConv3d(BasicQuantizedModule, nn.Conv3d):
    @classmethod
    def from_qat(cls, qat_module):
        return QuantizedConvNd._from_qat(cls, qat_module)

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
            weight_qparams: dict = {},) -> None:

        nn.Conv3d.__init__(
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
            dtype
        )
        self._init_weight_qparams(weight_qparams)

    def forward(self, x):
        assert self.padding_mode == 'zeros'

        weight = self.get_fake_quant_weight()
        result = F.conv3d(
            x,
            weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result
