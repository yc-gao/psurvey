import torch

from .module_quantizer import ModuleQuantizer


class QnnQuantizer:
    def __init__(self) -> None:
        super().__init__()

        self._quantizer = ModuleQuantizer()

    def quantize(graph_module: torch.fx.GraphModule):
        # TODO: yc
        return graph_module
