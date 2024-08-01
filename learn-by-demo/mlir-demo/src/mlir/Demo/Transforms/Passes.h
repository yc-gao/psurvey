#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace demo {

#define GEN_PASS_DECL
#include "mlir/Demo/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "mlir/Demo/Transforms/Passes.h.inc"

}  // namespace demo
}  // namespace mlir
