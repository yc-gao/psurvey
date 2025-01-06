import torch

from .convert_observer_or_fq import ConvertObserverOrFq
from .qat_converter import QatConverter

from .utils import partition_module_name, get_new_attr_name_with_prefix


class BasicQuantizer:
    # [
    #     {
    #         'module_name': '',
    #         'module_type': '',
    #         'activation': '',
    #         'weight': ''
    #     }
    # ]
    def quantize_modules(self, graph_module: torch.fx.GraphModule, qconfigs):
        module_name_to_qconfig = {
            qconfig['module_name']: qconfig for qconfig in qconfigs if 'module_name' in qconfig
        }
        module_type_to_qconfig = {
            qconfig['module_type']: qconfig for qconfig in qconfigs if 'module_type' in qconfig
        }
        for node in graph_module.graph.nodes:
            if node.op != "call_module":
                continue
            mod = graph_module.get_submodule(node.target)

            qconfig = module_name_to_qconfig.get(node.target, None)
            qconfig = qconfig or module_type_to_qconfig.get(type(mod), None)
            if qconfig is None:
                continue

            node.meta['qconfig'] = qconfig  # used in ConvertToQatModule

            act_qconfig = qconfig.get('activation', None)
            if act_qconfig is None:
                continue

            parent_name, _ = partition_module_name(node.target)
            parent_mod = graph_module.get_submodule(parent_name)

            fq_name = get_new_attr_name_with_prefix(parent_mod, "fq")
            setattr(parent_mod, fq_name, act_qconfig())

            with graph_module.graph.inserting_after(node):
                fq_node = graph_module.graph.create_node(
                    'call_module',
                    f"{parent_name}.{fq_name}" if parent_name else fq_name
                )
                node.replace_all_uses_with(fq_node)
                fq_node.args = (node,)
        graph_module = torch.fx.GraphModule(graph_module, graph_module.graph)

        graph_module = QatConverter.ConvertToQatModule(graph_module)
        return graph_module

    def finalize(self, graph_module: torch.fx.GraphModule):
        graph_module = ConvertObserverOrFq.apply(graph_module)
        graph_module = QatConverter.ConvertToQuantizedModule(graph_module)
        return graph_module
