from typing import Tuple
from typing import NamedTuple

from onnxutils.common import OnnxNode


class OnnxToTorchModule:
    pass


class OnnxMapping(NamedTuple):
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


class OperationDescription(NamedTuple):
    domain: str
    operation_type: str
    version: int


def onnx_mapping_from_node(node: OnnxNode) -> OnnxMapping:
    return OnnxMapping(
        inputs=node.inputs(),
        outputs=node.outputs(),
    )
