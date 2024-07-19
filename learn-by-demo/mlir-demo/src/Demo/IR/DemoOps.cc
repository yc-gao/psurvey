#include "DemoOps.h"

#include "DemoOpsDialect.cpp.inc"

void mlir::demo::DemoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DemoOps.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "DemoOps.cpp.inc"
