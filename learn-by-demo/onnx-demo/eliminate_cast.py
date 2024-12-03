from onnx_model import OnnxModel


class EliminateCast:
    @staticmethod
    def apply(onnx_model: OnnxModel) -> OnnxModel:
        name_to_dtype = {
            v.name: v.type.tensor_type.elem_type for v in onnx_model.vinfos() if v.type.HasField('tensor_type')
        }
        name_to_dtype.update({
            v.name: v.type.tensor_type.elem_type for v in onnx_model.inputs() if v.type.HasField('tensor_type')
        })
        name_to_dtype.update({
            v.name: v.type.tensor_type.elem_type for v in onnx_model.outputs() if v.type.HasField('tensor_type')
        })
        name_to_dtype.update({
            v.name: v.data_type for v in onnx_model.initializers()
        })

        input_name_map = {}
        for node in reversed(onnx_model.nodes()):
            if node.op_type == 'Cast':
                itype = name_to_dtype.get(node.input[0], None)
                otype = name_to_dtype.get(node.output[0], None)
                if itype and otype and otype == itype:
                    input_name_map[node.output[0]] = node.input[0]

        onnx_model.remap_input_names(input_name_map)
        onnx_model.remove_unused()
        return onnx_model
