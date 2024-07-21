#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace demo {

void AddPassesDemoToLLVM(mlir::PassManager&);

}  // namespace demo
}  // namespace mlir
