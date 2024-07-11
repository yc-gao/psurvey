#pragma once

#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace toy {

void addPassesToLLVM(mlir::PassManager&);

}  // namespace toy
}  // namespace mlir
