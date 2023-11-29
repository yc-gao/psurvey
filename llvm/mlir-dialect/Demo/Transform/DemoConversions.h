#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>
namespace mlir {
namespace demo {

std::unique_ptr<Pass> createLowerToAffinePass();

} // namespace demo
} // namespace mlir
