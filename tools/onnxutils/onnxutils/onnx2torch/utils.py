from typing import Tuple
from typing import NamedTuple

import torch
from onnxutils.onnx import OnnxNode


class OperationDescription(NamedTuple):
    domain: str
    operation_type: str
    version: int


class OnnxToTorchModule:
    pass


class OnnxMapping(NamedTuple):
    inputs: Tuple[str, ...]
    outputs: Tuple[str, ...]


class OperationConverterResult(NamedTuple):
    torch_module: torch.nn.Module
    onnx_mapping: OnnxMapping


def onnx_mapping_from_node(onnx_node: OnnxNode) -> OnnxMapping:
    return OnnxMapping(
        inputs=onnx_node.inputs(),
        outputs=onnx_node.outputs(),
    )
