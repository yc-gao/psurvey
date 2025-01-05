import torch
from torch.ao.quantization.fx.tracer import QuantizationTracer


def symbolic_trace(
        m,
        concrete_args=None,
        skipped_module_names=[],
        skipped_module_classes=[]):
    tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)
    graph = tracer.trace(m, concrete_args)
    return torch.fx.GraphModule(m, graph)
