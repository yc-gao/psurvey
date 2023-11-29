#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Types.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/raw_ostream.h"

#include "Demo.h"
#include "DemoConversions.h"

mlir::ModuleOp CreateModule(mlir::OpBuilder &builder) {
  auto module = mlir::ModuleOp::create(builder.getUnknownLoc());
  builder.setInsertionPointToEnd(module.getBody());

  auto func_type = builder.getFunctionType({}, {});
  auto func_op = builder.create<mlir::func::FuncOp>(builder.getUnknownLoc(),
                                                    "main", func_type);
  builder.setInsertionPointToStart(func_op.addEntryBlock());

  builder.create<mlir::arith::ConstantOp>(builder.getUnknownLoc(),
                                          builder.getF64Type(),
                                          builder.getF64FloatAttr(1));

  mlir::Type elem_type = builder.getF64Type();
  mlir::Type tensor_type = mlir::RankedTensorType::get({2}, elem_type);
  mlir::DenseElementsAttr attr =
      mlir::DenseElementsAttr::get(tensor_type, {1.L, 2.L});
  builder.create<mlir::demo::ConstantOp>(builder.getUnknownLoc(), tensor_type,
                                         attr);

  builder.create<mlir::func::ReturnOp>(builder.getUnknownLoc());

  return module;
}

int main(int argc, char *argv[]) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::demo::DemoDialect>();
  mlir::MLIRContext context(registry);
  context.loadAllAvailableDialects();

  mlir::OpBuilder builder(&context);

  auto module = CreateModule(builder);
  module.dump();

  mlir::PassManager pm(&context);
  pm.addPass(mlir::demo::createLowerToAffinePass());
  if (mlir::failed(pm.run(module))) {
    llvm::errs() << "run pm failed";
    return 1;
  }
  module.dump();
  return 0;
}
