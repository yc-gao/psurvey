#pragma once

#include "Demo/IR/DemoOps.h"
#include "mlir/IR/DialectRegistry.h"

namespace mlir {
namespace demo {

inline void registerDialects(mlir::DialectRegistry& registry) {
  registry.insert<mlir::demo::DemoDialect>();
}  // namespace demo

inline void registerDialects(mlir::MLIRContext& context) {
  mlir::DialectRegistry registry;
  registerDialects(registry);
  context.appendDialectRegistry(registry);
}  // namespace demo

}  // namespace demo
}  // namespace mlir
