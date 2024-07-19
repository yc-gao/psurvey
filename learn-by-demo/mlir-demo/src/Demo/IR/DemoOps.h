#pragma once

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "DemoOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "DemoOps.h.inc"
