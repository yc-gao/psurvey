from operator import getitem

import torch

from onnxutils.onnx import OnnxModel, OnnxNode

from .registry import find_converter
from .utils import OnnxMapping


class InitializersContainer(torch.nn.Module):
    def add_initializer(self, name: str, initializer: torch.Tensor) -> None:
        self.register_buffer(name, initializer)

    def forward(self, *args, **kwargs):
        raise RuntimeError('Got unexpected "forward" on constant container')


def normalize_module_name(name, domain='', op_type=''):
    return (f'{domain}/' + (name.replace('.', '/') or op_type)).lstrip('/')


def convert(
    onnx_model: OnnxModel,
    keep_input_names: bool = True,
) -> torch.fx.GraphModule:
    opset_import = {
        opsetid_proto.domain: opsetid_proto.version for opsetid_proto in onnx_model.opsets()}

    root_initializer = InitializersContainer()
    root_module = torch.nn.Module()
    root_module.add_module('initializers', root_initializer)
    torch_nodes = {}

    torch_graph = torch.fx.Graph()

    root_mapping = OnnxMapping(
        inputs=tuple(x for x in sorted(onnx_model.input_names())),
        outputs=tuple(x for x in sorted(onnx_model.output_names())),
    )

    # create input nodes
    for idx, name in enumerate(root_mapping.inputs, 1):
        if keep_input_names:
            if not name.isidentifier():
                raise ValueError(
                    f'Input name "{name}" cannot be used as name of placeholder in fx.GraphModule.')
            placeholder_name = name
        else:
            placeholder_name = f'input_{idx}'
        torch_nodes[name] = torch_graph.placeholder(name=placeholder_name)

    for onnx_node in onnx_model.nodes():
        version = opset_import[onnx_node.domain()]
        converter = find_converter(
            domain=onnx_node.domain(),
            operation_type=onnx_node.op_type(),
            version=version,
        )

        torch_module, onnx_mapping = converter(onnx_node, onnx_model)
        setattr(torch_module, 'onnx_mapping', onnx_mapping)
        root_module.add_module(
            normalize_module_name(onnx_node.name()),
            torch_module
        )

        args = []
        for value_name in onnx_mapping.inputs:
            if onnx_model.get_input_by_name(value_name) is not None:
                args.append(torch_nodes[value_name])
            elif onnx_model.get_initializer_by_name(value_name) is not None:
                if value_name not in torch_nodes:
                    buffer_idx = sum(
                        1 for _ in root_initializer.buffers())
                    buffer_name = f'onnx_initializer_{buffer_idx}'
                    root_initializer.add_initializer(
                        buffer_name,
                        onnx_model.get_initializer_by_name(
                            value_name).to_torch(),
                    )
                    torch_nodes[value_name] = torch_graph.get_attr(
                        f'initializers.{buffer_name}')
                args.append(torch_nodes[value_name])

            elif onnx_model.get_node_by_output(value_name):
                onnx_input_node: OnnxNode = onnx_model.get_node_by_output(
                    value_name)
                torch_input_node = torch_nodes[onnx_input_node.name()]
                if len(onnx_input_node.outputs()) > 1:
                    index = onnx_input_node.outputs().index(value_name)
                    torch_input_node = torch_graph.call_function(
                        getitem, args=(torch_input_node, index))
                args.append(torch_input_node)

            else:
                raise RuntimeError(
                    f'Got unexpected input value type ({value_name})')

        torch_nodes[onnx_node.name()] = torch_graph.call_module(
            module_name=normalize_module_name(onnx_node.name()), args=tuple(args))

    if len(root_mapping.outputs) > 1:
        torch_graph.output(
            tuple(
                torch_nodes[onnx_model.get_node_by_output(output_name).name()]
                for output_name in root_mapping.outputs
            ))
    else:
        output_name = root_mapping.outputs[0]
        torch_graph.output(
            torch_nodes[onnx_model.get_node_by_output(output_name).name()]
        )

    torch_graph.lint()
    torch_model = torch.fx.GraphModule(root=root_module, graph=torch_graph)
    setattr(torch_model,
            'onnx_mapping',
            root_mapping
            )
    return torch_model
