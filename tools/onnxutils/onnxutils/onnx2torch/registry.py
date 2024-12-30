from onnx import defs

from .utils import OperationDescription


_CONVERTER_REGISTRY = {}


def converter(
    operation_type: str,
    version: int,
    domain: str = defs.ONNX_DOMAIN,
):
    def deco(converter):
        description = OperationDescription(
            domain=domain,
            operation_type=operation_type,
            version=version,
        )
        if description in _CONVERTER_REGISTRY:
            raise ValueError(f'Operation "{description}" already registered')
        _CONVERTER_REGISTRY[description] = converter
        return converter
    return deco


def find_converter(
    operation_type: str,
    version: int,
    domain: str = defs.ONNX_DOMAIN,
):
    try:
        version = defs.get_schema(
            operation_type,
            domain=domain,
            max_inclusive_version=version,
        ).since_version
    except (RuntimeError, defs.SchemaError):
        pass

    description = OperationDescription(
        domain=domain,
        operation_type=operation_type,
        version=version,
    )

    converter = _CONVERTER_REGISTRY.get(description, None)
    if converter is None:
        raise NotImplementedError(
            f'Converter is not implemented ({description})')

    return converter
