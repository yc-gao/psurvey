#include "Demo.h"

#include "DemoDialect.cpp.inc"

namespace mlir {
namespace demo {

void DemoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "DemoOps.cpp.inc"
      >();
}

} // namespace demo
} // namespace mlir

#define GET_OP_CLASSES
#include "DemoOps.cpp.inc"
