#ifndef MLIR_DIALECT_DEMO_TRANSFORMS_PASSES
#define MLIR_DIALECT_DEMO_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def DemoNormalizePass : Pass<"demo-normalize", "ModuleOp"> {}

#endif
