import torch
import torch.nn as nn

from torch.ao.quantization.fake_quantize import FakeQuantize
from torch.ao.quantization.observer import HistogramObserver, PerChannelMinMaxObserver

from onnxutils.quantization.quantizer import BasicQuantizer
from onnxutils.quantization.utils import graph_get_module


class QnnQuantizer(BasicQuantizer):
    _conv_layers = (
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
    )

    _bn_layers = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
    )

    def __init__(self) -> None:
        super().__init__()

    def quantize(self, graph_module: torch.fx.GraphModule):
        qconfig_mapping = {}

        for node in graph_module.graph.nodes:
            if graph_get_module(graph_module, node, QnnQuantizer._conv_layers) is not None:
                prev_node = node.args[0]
                prev_qconfig = qconfig_mapping.get(prev_node.name, None)
                if prev_qconfig is None:
                    qconfig_mapping[prev_node.name] = {
                        'activation': FakeQuantize.with_args(observer=HistogramObserver)
                    }

                qconfig_mapping[node.name] = {
                    'weight': FakeQuantize.with_args(
                        observer=PerChannelMinMaxObserver
                    ),
                    'activation': FakeQuantize.with_args(observer=HistogramObserver)
                }
            elif graph_get_module(graph_module, node, QnnQuantizer._bn_layers) is not None:
                prev_node = node.args[0]
                if graph_get_module(graph_module, prev_node, QnnQuantizer._conv_layers) is None:
                    continue
                prev_qconfig = qconfig_mapping.get(prev_node.name, None)
                if prev_qconfig is None:
                    continue
                qconfig_mapping[node.name] = {
                    'activation': prev_qconfig.pop('activation')
                }
            elif graph_get_module(graph_module, node, nn.ReLU):
                prev_node = node.args[0]
                if graph_get_module(graph_module, prev_node, QnnQuantizer._conv_layers + QnnQuantizer._bn_layers) is None:
                    continue
                prev_qconfig = qconfig_mapping.get(prev_node.name, None)
                if prev_qconfig is None:
                    continue
                qconfig_mapping[node.name] = {
                    'activation': prev_qconfig.pop('activation')
                }
        qconfigs = [
            v | {'name': k} for k, v in qconfig_mapping.items()
        ]
        return super().quantize(graph_module, qconfigs)
