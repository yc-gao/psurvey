import torch

from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.fake_quantize import FakeQuantizeBase


class NodeQuantizer:
    @staticmethod
    def get_new_attr_name_with_prefix(
            module: torch.nn.Module,
            prefix: str,
            idx: int = 0):
        prefix = prefix.replace(".", "_")

        while True:
            attr_name = f"{prefix}{idx}"
            if not hasattr(module, attr_name):
                break
            idx += 1
        return attr_name

    @staticmethod
    def partition_module_name(target: str):
        r = target.rsplit('.', 1)
        if len(r) == 1:
            return '', r[0]
        return r[0], r[1]

    def quantize(
            self,
            graph_module: torch.fx.GraphModule,
            qconfigs: list[dict]):
        qconfig_mapping: dict[str, dict] = {
            qconfig['name']: qconfig for qconfig in qconfigs
        }

        for node in graph_module.graph.nodes:
            qconfig = qconfig_mapping.get(node.name, None)
            if qconfig is None:
                continue

            act_qconfig = qconfig.get('activation', None)
            if act_qconfig is None:
                continue

            fq_name = self.get_new_attr_name_with_prefix(
                graph_module, 'fq')
            graph_module.add_submodule(fq_name, act_qconfig())

            with graph_module.graph.inserting_after(node):
                fq_node = graph_module.graph.create_node(
                    'call_module',
                    fq_name
                )
                node.replace_all_uses_with(fq_node)
                fq_node.args = (node,)

        return torch.fx.GraphModule(graph_module, graph_module.graph)

    def finalize(
        self,
        graph_module: torch.fx.GraphModule
    ):
        for node in graph_module.graph.nodes:
            if node.op != 'call_module':
                continue

            mod = graph_module.get_submodule(node.target)
            if not isinstance(mod, (ObserverBase, FakeQuantizeBase)):
                continue

            qscheme = mod.qscheme
            quant_min = mod.quant_min
            quant_max = mod.quant_max
            scale, zero_point = mod.calculate_qparams()

            qparams = {
                'qscheme': qscheme,
                'quant_min': quant_min,
                'quant_max': quant_max,
                'scale': scale.to(torch.float),
                'zero_point': zero_point.to(torch.int32),
            }

            if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
                op_func = torch.fake_quantize_per_tensor_affine
            elif qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
                op_func = torch.fake_quantize_per_channel_affine
                qparams['ch_axis'] = mod.ch_axis
            else:
                raise NotImplementedError

            device = next(iter(graph_module.parameters())).device
            parent_name, _ = self.partition_module_name(node.target)
            parent_mod = graph_module.get_submodule(parent_name)

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
                        attr_name = self.get_new_attr_name_with_prefix(
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
