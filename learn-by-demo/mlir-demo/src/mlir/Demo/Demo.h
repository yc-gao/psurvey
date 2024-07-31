#pragma once

#include "mlir/Demo/IR/DemoOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace demo {

inline void registerDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::demo::DemoDialect>();
}

inline void registerDialects(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registerDialects(registry);
  context.appendDialectRegistry(registry);
}

inline void AddPasses(mlir::PassManager& pm) {}

}  // namespace demo
}  // namespace mlir
