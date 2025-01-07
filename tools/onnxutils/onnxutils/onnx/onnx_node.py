from onnx.onnx_ml_pb2 import NodeProto, AttributeProto

from .onnx_tensor import OnnxTensor


class OnnxNode:
    @staticmethod
    def _parse_attribute_value(attribute: AttributeProto):
        if attribute.HasField('i'):
            value = attribute.i
        elif attribute.HasField('f'):
            value = attribute.f
        elif attribute.HasField('s'):
            value = str(attribute.s, 'utf-8')
        elif attribute.HasField('t'):
            value = OnnxTensor(attribute.t)
        elif attribute.ints:
            value = list(attribute.ints)
        elif attribute.floats:
            value = list(attribute.floats)
        elif attribute.strings:
            value = [str(s, 'utf-8') for s in attribute.strings]
        elif attribute.tensors:
            value = [OnnxTensor(t) for t in attribute.tensors]
        else:
            value = attribute
        return value

    def clone(self):
        t = NodeProto()
        t.CopyFrom(self._proto)
        return OnnxNode(t)

    def __init__(self, onnx_node: NodeProto):
        self._proto = onnx_node

        self._inputs = tuple(self._proto.input)
        self._outputs = tuple(self._proto.output)

    def proto(self):
        return self._proto

    def name(self):
        return self._proto.name

    def domain(self):
        return self._proto.domain

    def op_type(self):
        return self._proto.op_type

    def inputs(self):
        return self._inputs

    def outputs(self):
        return self._outputs

    def attributes(self):
        return {
            attribute.name: OnnxNode._parse_attribute_value(attribute)
            for attribute in self._proto.attribute
        }
