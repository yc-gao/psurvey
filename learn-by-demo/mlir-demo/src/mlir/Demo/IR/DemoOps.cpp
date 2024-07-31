#include "DemoOps.h"

#include "mlir/Demo/IR/DemoOpsDialect.cpp.inc"

void mlir::demo::DemoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Demo/IR/DemoOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Demo/IR/DemoOps.cpp.inc"
