#pragma once

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "mlir/Demo/IR/DemoOpsDialect.h.inc"

#define GET_OP_CLASSES
#include "mlir/Demo/IR/DemoOps.h.inc"
