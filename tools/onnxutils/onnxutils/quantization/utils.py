import torch
from torch.ao.quantization.fx.tracer import QuantizationTracer


def get_new_attr_name_with_prefix(module, prefix, idx=0):
    prefix = prefix.replace(".", "_")

    attr_name = f"{prefix}{idx}"
    while hasattr(module, attr_name):
        idx += 1
        attr_name = f"{prefix}{idx}"

    return attr_name


def partition_module_name(target):
    r = target.rsplit('.', 1)
    if len(r) == 1:
        return '', r[0]
    return r[0], r[1]


def symbolic_trace(
        m,
        concrete_args=None,
        skipped_module_names=[],
        skipped_module_classes=[]):
    tracer = QuantizationTracer(skipped_module_names, skipped_module_classes)
    graph = tracer.trace(m, concrete_args)
    return torch.fx.GraphModule(m, graph)
