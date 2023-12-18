#pragma once
#include <memory>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

#include "demo/IR/DemoDialect.h"

namespace mlir {
namespace demo {

#define GEN_PASS_DECL
#include "Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Passes.h.inc"

} // namespace demo
} // namespace mlir
