#pragma once

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace toy {

std::unique_ptr<mlir::Pass> createToyConstantOpLoweringPass();

}
}  // namespace mlir
