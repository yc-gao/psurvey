import torch

from torch.ao.quantization.observer import ObserverBase
from torch.ao.quantization.fake_quantize import FakeQuantizeBase

from .convs import QuantizedConv2d


class BasicQuantizer:
    _quantized_module_mapping = {
        torch.ao.nn.qat.modules.conv.Conv2d: QuantizedConv2d
    }

    @staticmethod
    def get_new_attr_name_with_prefix(module, prefix, idx=0):
        prefix = prefix.replace(".", "_")

        attr_name = f"{prefix}{idx}"
        while hasattr(module, attr_name):
            idx += 1
            attr_name = f"{prefix}{idx}"

        return attr_name

    @staticmethod
    def partition_module_name(target):
        r = target.rsplit('.', 1)
        if len(r) == 1:
            return '', r[0]
        return r[0], r[1]

    def convert_observer_or_fq_to_qdq(self, graph_module: torch.fx.GraphModule, node: torch.fx.Node):
        mod = graph_module.get_submodule(node.target)
        scale, zero_point = mod.calculate_qparams()

        if mod.qscheme in (torch.per_channel_affine,
                           torch.per_channel_affine_float_qparams,
                           torch.per_channel_symmetric,):
            qparams = {
                "_scale_": scale,
                "_zero_point_": zero_point,
                "_axis_": int(mod.ch_axis),
                "_dtype_": mod.dtype,
            }
            quantize_op = torch.quantize_per_channel
        else:
            qparams = {
                "_scale_": float(scale),
                "_zero_point_": int(zero_point),
                "_dtype_": mod.dtype
            }
            quantize_op = torch.quantize_per_tensor

        parent_name, _ = BasicQuantizer.partition_module_name(node.target)
        parent_mod = graph_module.get_submodule(parent_name)

        with graph_module.graph.inserting_before(node):
            quantize_op_inputs = [node.args[0]]
            for key in ['_scale_', '_zero_point_',  '_axis_', '_dtype_']:
                value = qparams.get(key, None)
                if value is None:
                    continue
                if key in ["_scale_", "_zero_point_"]:
                    new_value = (
                        value.detach().clone()
                        if isinstance(value, torch.Tensor)
                        else torch.tensor(value, device=next(iter(graph_module.parameters())).device)
                    )
                    attr_name = BasicQuantizer.get_new_attr_name_with_prefix(
                        parent_mod,
                        key)
                    parent_mod.register_buffer(attr_name, new_value)
                    attr_node = graph_module.graph.create_node(
                        "get_attr",
                        f"{parent_name}.{attr_name}" if parent_name else attr_name
                    )
                    quantize_op_inputs.append(attr_node)
                else:
                    quantize_op_inputs.append(value)
            quantized_node = graph_module.graph.create_node(
                "call_function",
                quantize_op,
                tuple(quantize_op_inputs), {}
            )
            dequantized_node = graph_module.graph.call_method(
                "dequantize", args=(quantized_node,))
            node.replace_all_uses_with(dequantized_node)
            graph_module.graph.erase_node(node)

    def convert_quantized_module(self, graph_module: torch.fx.GraphModule, node: torch.fx.Node):
        mod = graph_module.get_submodule(node.target)
        observer_or_fake_quant = mod.weight_fake_quant

        qscheme = observer_or_fake_quant.qscheme
        if qscheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
            qscheme = torch.per_tensor_affine
        elif qscheme in (torch.per_channel_affine, torch.per_channel_symmetric):
            qscheme = torch.per_channel_affine
        else:
            raise ValueError

        dtype = observer_or_fake_quant.dtype
        scale, zero_point = observer_or_fake_quant.calculate_qparams()

        qparams = {
            'qscheme': qscheme,
            'dtype': dtype,
            'scale': scale,
            'zero_point': zero_point,
        }

        if qscheme == torch.per_channel_affine:
            qparams['axis'] = observer_or_fake_quant.ch_axis

        quantized_mod = self._quantized_module_mapping[type(mod)].from_float(
            mod.to_float(),
            qparams)
        parent_name, name = BasicQuantizer.partition_module_name(node.target)
        setattr(graph_module.get_submodule(parent_name), name, quantized_mod)

    def finalize(self, graph_module: torch.fx.GraphModule):
        for node in graph_module.graph.nodes:
            if node.op == 'call_module':
                mod = graph_module.get_submodule(node.target)
                if isinstance(mod, (ObserverBase, FakeQuantizeBase)):
                    self.convert_observer_or_fq_to_qdq(
                        graph_module,
                        node)
                elif type(mod) in self._quantized_module_mapping:
                    self.convert_quantized_module(graph_module, node)

        return torch.fx.GraphModule(graph_module, graph_module.graph)
