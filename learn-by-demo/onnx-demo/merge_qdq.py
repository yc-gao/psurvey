from onnx_model import OnnxModel


class MergeQdq:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        output_name_to_node = {
            output: node for node in onnx_model.nodes() for output in node.output
        }

        input_name_map = {}

        node_merged = set()
        for node in reversed(onnx_model.nodes()):
            if node.name in node_merged:
                continue
            if node.op_type == 'DequantizeLinear':
                inode = output_name_to_node.get(node.input[0], None)
                if inode and inode.op_type == 'QuantizeLinear':
                    input_name_map[node.output[0]] = inode.input[0]
                    node_merged.add(node.name)
                    node_merged.add(inode.name)
            elif node.op_type == 'QuantizeLinear':
                inode = output_name_to_node.get(node.input[0], None)
                if inode and inode.op_type == 'DequantizeLinear':
                    input_name_map[node.output[0]] = inode.input[0]
                    node_merged.add(node.name)
                    node_merged.add(inode.name)

        onnx_model.remap_input_names(input_name_map)
        onnx_model.remove_unused()
        return onnx_model
