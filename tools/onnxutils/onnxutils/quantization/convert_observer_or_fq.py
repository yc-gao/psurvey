import torch

from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from .utils import partition_module_name, get_new_attr_name_with_prefix


class ConvertObserverOrFq:
    @staticmethod
    def apply(graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op != 'call_module':
                continue
            mod = graph_module.get_submodule(node.target)
            if not isinstance(mod, (ObserverBase, FakeQuantizeBase)):
                continue

            device = next(iter(graph_module.parameters())).device
            parent_name, _ = partition_module_name(node.target)
            parent_mod = graph_module.get_submodule(parent_name)

            qscheme = mod.qscheme
            quant_min = mod.quant_min
            quant_max = mod.quant_max
            scale, zero_point = mod.calculate_qparams()

            qparams = {
                'qscheme': qscheme,
                'quant_min': quant_min,
                'quant_max': quant_max,
                'scale': scale,
                'zero_point': zero_point,
            }

            if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                op_func = torch.fake_quantize_per_tensor_affine
            elif qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
                op_func = torch.fake_quantize_per_channel_affine
                qparams['ch_axis'] = mod.ch_axis
            else:
                raise NotImplementedError

            with graph_module.graph.inserting_before(node):
                op_args = [node.args[0]]
                for name in ['scale', 'zero_point',  'ch_axis', 'quant_min', 'quant_max']:
                    if qparams.get(name, None) is None:
                        continue
                    if name in ('scale', 'zero_point'):
                        new_value = (
                            qparams[name].detach().clone()
                            if isinstance(qparams[name], torch.Tensor)
                            else torch.tensor(qparams[name], device=device)
                        )
                        attr_name = get_new_attr_name_with_prefix(
                            parent_mod, name)
                        parent_mod.register_buffer(attr_name, new_value)
                        attr_node = graph_module.graph.create_node(
                            'get_attr',
                            f"{parent_name}.{attr_name}"
                            if parent_name else attr_name
                        )
                        op_args.append(attr_node)
                    else:
                        op_args.append(qparams[name])
                op_node = graph_module.graph.create_node(
                    'call_function',
                    op_func,
                )
                node.replace_all_uses_with(op_node)
                graph_module.graph.erase_node(node)
                op_node.args = tuple(op_args)

        return torch.fx.GraphModule(graph_module, graph_module.graph)
