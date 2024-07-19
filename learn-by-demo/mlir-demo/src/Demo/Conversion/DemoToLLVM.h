#pragma once

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace demo {

std::unique_ptr<mlir::Pass> CreateDemoToLLVMPass();

}  // namespace demo
}  // namespace mlir
