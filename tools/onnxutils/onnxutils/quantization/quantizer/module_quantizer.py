import torch
import torch.nn as nn
from torch.ao.quantization import QConfig
import torch.ao.nn.qat as nnqat

from ..quantized_modules import QuantizedLinear, QuantizedConv1d, QuantizedConv2d, QuantizedConv3d
from .node_quantizer import NodeQuantizer


class ModuleQuantizer(NodeQuantizer):
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

    def quantize(
            self,
            graph_module: torch.fx.GraphModule,
            qconfigs: list[dict]):
        module_name_mapping = {
            node.target: node.name
            for node in graph_module.graph.nodes if node.op == 'call_module'
        }

        qconfigs = [
            x
            if 'name' in x
            else {'name': module_name_mapping[x['module_name']]} | x
            for x in qconfigs]

        graph_module = super().quantize(
            graph_module,
            qconfigs,
        )

        qconfig_mapping: dict[str, dict] = {
            qconfig['name']: qconfig for qconfig in qconfigs
        }

        for node in graph_module.graph.nodes:
            qconfig = qconfig_mapping.get(node.name, None)
            if qconfig is None:
                continue

            weight_qconfig = qconfig.get('weight', None)
            if weight_qconfig is None:
                continue

            mod = graph_module.get_submodule(node.target)
            mod.qconfig = QConfig(activation=None, weight=qconfig['weight'])
            new_mod = self._qat_module_mapping[
                type(mod)].from_float(mod)

            parent_name, name = self.partition_module_name(node.target)
            parent_module = graph_module.get_submodule(parent_name)
            setattr(parent_module, name, new_mod)
        return graph_module

    def finalize(
        self,
        graph_module: torch.fx.GraphModule
    ):
        graph_module = super().finalize(graph_module)

        for node in graph_module.graph.nodes:
            if node.op != 'call_module':
                continue

            mod = graph_module.get_submodule(node.target)
            if type(mod) not in self._quantized_module_mapping:
                continue

            new_mod = self._quantized_module_mapping[
                type(mod)].from_qat(mod)
            parent_name, name = self.partition_module_name(node.target)
            parent_module = graph_module.get_submodule(parent_name)
            setattr(parent_module, name, new_mod)

        return graph_module
