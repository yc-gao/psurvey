import torch
from torch import nn

from torch.ao.quantization import QConfig

from .quantizer import BasicQuantizer


class QnnQuantizer(BasicQuantizer):
    def __init__(self) -> None:
        super().__init__()

    def get_module(self, graph_module, node):
        if node.op != 'call_module':
            return None
        return graph_module.get_submodule(node.target)

    def annotate_module_names(self, graph_module, qconfigs: dict):
        for node in graph_module.graph.nodes:
            if 'qconfig' in node.meta:
                continue
            if node.op == 'call_module' and node.target in qconfigs:
                node.meta['qconfig'] = qconfigs[node.target]

    def annotate_module_types(self, graph_module, qconfigs: dict):
        for node in graph_module.graph.nodes:
            if 'qconfig' in node.meta:
                continue
            if node.op == 'call_module':
                mod = graph_module.get_submodule(node.target)
                if type(mod) in qconfigs:
                    node.meta['qconfig'] = qconfigs[type(mod)]
    # [
    #     {
    #         'name': '',
    #         'weight': {}
    #         'input': {}
    #         'output': {}
    #     },
    #     {
    #         'type': '',
    #         'weight': {}
    #         'input': {}
    #         'output': {}
    #     },
    # ]

    def annotate_qconfigs(self, graph_module, qconfigs):
        module_name_qconfigs = {
            qconfig['name']: qconfig
            for qconfig in qconfigs if 'name' in qconfig
        }
        self.annotate_module_names(graph_module, module_name_qconfigs)

        module_type_qconfigs = {
            qconfig['type']: qconfig
            for qconfig in qconfigs if 'type' in qconfig
        }
        self.annotate_module_types(graph_module, module_type_qconfigs)

    def normlize_annotattion(self, graph_module):
        for node in graph_module.graph.nodes:
            qconfig = node.meta.get('qconfig', None)
            if qconfig is None:
                continue
            if 'input' in qconfig:
                for arg_node in node.args:
                    arg_qconfig = arg_node.meta.get('qconfig', None) or {}
                    if 'output' not in arg_qconfig:
                        arg_qconfig['output'] = qconfig['input']
                    arg_node.meta['qconfig'] = arg_qconfig
            qconfig.pop('input')
            node.meta['qconfig'] = qconfig

    # def fuse_conv_relu(self, graph_module):
    #     for maybe_relu_node in graph_module.graph.nodes:
    #         if 'qconfig' not in maybe_relu_node.meta:
    #             continue
    #         maybe_relu_mod = self.get_module(graph_module, maybe_relu_node)
    #         if maybe_relu_mod is None or not isinstance(maybe_relu_mod, nn.ReLU):
    #             continue
    #         maybe_conv_node = maybe_relu_node.args[0]
    #
    #         if 'qconfig' not in maybe_conv_node.meta:
    #             continue
    #         maybe_conv_mod = self.get_module(graph_module, maybe_conv_node)
    #         if maybe_conv_mod is None or not isinstance(maybe_conv_mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
    #             continue
    #         if len(maybe_conv_node.users) > 1:
    #             continue
    #
    #         if 'input' in maybe_relu_node.meta['qconfig']:
    #             del maybe_relu_node.meta['qconfig']['input']
    #
    # def fuse_conv_bn_relu(self, graph_module):
    #     for maybe_relu_node in graph_module.graph.nodes:
    #         if 'qconfig' not in maybe_relu_node.meta:
    #             continue
    #         maybe_relu_mod = self.get_module(graph_module, maybe_relu_node)
    #         if not maybe_relu_mod or not isinstance(maybe_relu_mod, nn.ReLU):
    #             continue
    #         maybe_bn_node = maybe_relu_node.args[0]
    #
    #         if 'qconfig' not in maybe_bn_node.meta:
    #             continue
    #         maybe_bn_mod = self.get_module(graph_module, maybe_bn_node)
    #         if not maybe_bn_mod or not isinstance(maybe_bn_mod, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
    #             continue
    #         if len(maybe_bn_node.users) > 1:
    #             continue
    #         maybe_conv_node = maybe_bn_mod.args[0]
    #
    #         if 'qconfig' not in maybe_conv_node.meta:
    #             continue
    #         maybe_conv_mod = self.get_module(graph_module, maybe_conv_node)
    #         if not maybe_conv_mod or not isinstance(maybe_conv_mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
    #             continue
    #         if len(maybe_conv_node.users) > 1:
    #             continue
    #
    #         if 'input' in maybe_relu_node.meta['qconfig']:
    #             del maybe_relu_node.meta['qconfig']['input']
    #         if 'input' in maybe_bn_node.meta['qconfig']:
    #             del maybe_bn_node.meta['qconfig']['input']

    def append_observer_or_fq(self, graph_module, node, observer_or_fq):
        parent_name, _ = self.partition_module_name(node.target)
        parent_mod = graph_module.get_submodule(parent_name)

        attr_name = self.get_new_attr_name_with_prefix(
            parent_mod,
            "fq")
        setattr(parent_mod, attr_name, observer_or_fq)

        with graph_module.graph.inserting_after(node):
            observer_or_fq_node = graph_module.graph.call_module(
                f"{parent_name}.{attr_name}" if parent_name else attr_name
            )
            node.replace_all_uses_with(observer_or_fq_node)
            observer_or_fq_node.args = (node,)

    def apply_quantization(self, graph_module):
        for node in graph_module.graph.nodes:
            qconfig = node.meta.get('qconfig', None)
            if qconfig is None:
                continue
            if 'output' in qconfig:
                self.append_observer_or_fq(
                    graph_module, node, qconfig['output']())

            if 'weight' in qconfig:
                assert node.op == 'call_module'
                mod = graph_module.get_submodule(node.target)
                mod.qconfig = QConfig(
                    activation=None,
                    weight=qconfig['weight']
                )
                new_mod = self._unquantized_module_mapping[
                    type(mod)].from_float(mod)

                parent_name, name = self.partition_module_name(node.target)
                parent_mod = graph_module.get_submodule(parent_name)
                setattr(parent_mod, name, new_mod)

    def quantize(self, graph_module, qconfigs):
        self.annotate_qconfigs(graph_module, qconfigs)
        self.normlize_annotattion(graph_module)
        # self.fuse_conv_bn_relu(graph_module)
        # self.fuse_conv_relu(graph_module)
        self.apply_quantization(graph_module)
        return torch.fx.GraphModule(graph_module, graph_module.graph)
