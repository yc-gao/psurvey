#include "DemoDialect.h"
#include "DemoOps.h"

namespace mlir {
namespace demo {

void DemoDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "demo/IR/DemoOps.cpp.inc"
      >();
}

} // namespace demo
} // namespace mlir

#include "demo/IR/DemoDialect.cpp.inc"
