#ifndef DEMO_PASSES
#define DEMO_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDemoToArith : Pass<"convert-demo-to-arith", "mlir::func::FuncOp"> {
    let dependentDialects = ["mlir::arith::ArithDialect", "mlir::demo::DemoDialect"];
}

#endif
