#pragma once
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "DemoDialect.h.inc"

#define GET_OP_CLASSES
#include "DemoOps.h.inc"
