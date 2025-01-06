import torch
import torch.nn as nn
import torch.ao.nn.qat as nnqat
from torch.ao.quantization import QConfig

from .utils import partition_module_name
from .quantized_modules import QuantizedLinear, QuantizedConv1d, QuantizedConv2d, QuantizedConv3d


class QatConverter:
    _qat_module_mapping = {
        nn.modules.Linear: nnqat.modules.Linear,

        nn.modules.Conv1d: nnqat.modules.Conv1d,
        nn.modules.Conv2d: nnqat.modules.Conv2d,
        nn.modules.Conv3d: nnqat.modules.Conv3d,
    }
    _quantized_module_mapping = {
        nnqat.modules.Linear: QuantizedLinear,

        nnqat.modules.Conv1d: QuantizedConv1d,
        nnqat.modules.Conv2d: QuantizedConv2d,
        nnqat.modules.Conv3d: QuantizedConv3d,
    }

    @staticmethod
    def ConvertToQatModule(graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            qconfig = node.meta.get('qconfig', None)
            if qconfig is None or 'weight' not in qconfig:
                continue
            assert node.op == 'call_module'

            mod = graph_module.get_submodule(node.target)
            if type(mod) not in QatConverter._qat_module_mapping:
                continue

            parent_name, name = partition_module_name(node.target)
            parent_module = graph_module.get_submodule(parent_name)

            mod.qconfig = QConfig(activation=None, weight=qconfig['weight'])
            new_mod = QatConverter._qat_module_mapping[
                type(mod)].from_float(mod)
            setattr(parent_module, name, new_mod)
        return graph_module

    @staticmethod
    def ConvertToQuantizedModule(graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            qconfig = node.meta.get('qconfig', None)
            if qconfig is None or 'weight' not in qconfig:
                continue
            assert node.op == 'call_module'

            mod = graph_module.get_submodule(node.target)
            if type(mod) not in QatConverter._quantized_module_mapping:
                continue

            parent_name, name = partition_module_name(node.target)
            parent_module = graph_module.get_submodule(parent_name)

            new_mod = QatConverter._quantized_module_mapping[
                type(mod)].from_qat(mod)
            setattr(parent_module, name, new_mod)
        return graph_module
