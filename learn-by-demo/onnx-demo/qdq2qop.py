#!/usr/bin/env python3
import argparse

import onnx

from onnx_model import OnnxModel


class QdqFinalizer:
    FinalizerRegistry = {}

    def __init__(self, model):
        self.model = OnnxModel(model)

    @staticmethod
    def Finalizer(op_type):
        def wrapper(cls):
            assert op_type not in QdqFinalizer.FinalizerRegistry
            QdqFinalizer.FinalizerRegistry[op_type] = cls
            return cls
        return wrapper

    @staticmethod
    def DoFinalize(node, input_name_to_q, output_name_to_dq):
        op_finalizer = QdqFinalizer.FinalizerRegistry.get(node.op_type, None)
        if op_finalizer:
            return op_finalizer.Finalize(node, input_name_to_q, output_name_to_dq)
        return [], [], [], {}

    def Qdq2Qop(self):
        output_name_to_dq = {
            node.output[0]: node for node in self.model.nodes() if node.op_type == 'DequantizeLinear'
        }
        input_name_to_q = {
            node.input[0]: node for node in self.model.nodes() if node.op_type == 'QuantizeLinear'
        }

        nodes_merged_set = set()
        new_nodes = []
        new_initializers = []
        io_maps = {}
        for node in self.model.nodes():
            new_nodes.append(node)
            output_nodes, initializers, merged_nodes, io_map = QdqFinalizer.DoFinalize(
                node, input_name_to_q, output_name_to_dq)
            nodes_merged_set.update([n.name for n in merged_nodes])
            new_nodes.extend(output_nodes)
            new_initializers.extend(initializers)
            io_maps.update(io_map)

        self.model.clear_nodes()
        self.model.add_nodes(
            [node for node in new_nodes if node.name not in nodes_merged_set])
        self.model.add_initializers(new_initializers)
        self.model.remap_names(io_maps)
        self.model.topological_sort()
        self.model.remove_unused()

    def EliminateQdq(self):
        # TODO: yinchao
        pass

    def Finalize(self):
        self.Qdq2Qop()
        self.EliminateQdq()


@QdqFinalizer.Finalizer('Conv')
class ConvFinalizer:
    @staticmethod
    def Finalize(node, input_name_to_qdq, output_name_to_qdq):
        dq_nodes = [output_name_to_qdq.get(n, None) for n in node.input]
        q_node,  = [input_name_to_qdq.get(n, None) for n in node.output]

        if None in dq_nodes or q_node is None:
            return [], [], [], {}

        input_names = []
        for dq_node in dq_nodes[:2]:
            input_names.extend(dq_node.input)
        input_names.extend(q_node.input[1:])
        if len(dq_nodes) > 2:
            # ref: https://onnx.ai/onnx/operators/onnx__QLinearConv.html
            # scale == x_scale * w_scale and zero_point == 0
            input_names.append(dq_nodes[2].input[0])

        output_names = [node.name + '_quantized']

        new_node = onnx.helper.make_node(
            'QLinearConv',
            input_names,
            output_names,
            node.name + '_quant',
        )
        # TODO:
        new_node.attribute.extend(node.attribute)
        return [new_node], [], dq_nodes + [node, q_node], {k.output[0]: v for k, v in zip([q_node], output_names)}


def parse_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', type=str, default='output.onnx')
    parser.add_argument('model')
    return parser.parse_args()


def main():
    options = parse_options()

    finalizer = QdqFinalizer(options.model)
    finalizer.Finalize()

    if options.output:
        onnx.save(finalizer.model.model, options.output)


if __name__ == "__main__":
    main()
