#include "Passes.h"

#include "demo/IR/DemoDialect.h"

namespace mlir {
namespace demo {

#define GEN_PASS_DEF_CONVERTDEMOTOARITH
#include "Passes.h.inc"

namespace {

struct ConvertDemoToArith
    : public impl::ConvertDemoToArithBase<ConvertDemoToArith> {
  void runOnOperation() override {}
};

} // namespace

} // namespace demo
} // namespace mlir
