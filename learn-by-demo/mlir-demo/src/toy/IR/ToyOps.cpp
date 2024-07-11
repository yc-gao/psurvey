#include "ToyOps.h"

#include "ToyOpsDialect.cpp.inc"

namespace mlir {
namespace toy {

void ToyDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "ToyOps.cpp.inc"
      >();
}
}  // namespace toy
}  // namespace mlir

#define GET_OP_CLASSES
#include "ToyOps.cpp.inc"
