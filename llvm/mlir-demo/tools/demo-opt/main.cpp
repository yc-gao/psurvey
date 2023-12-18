#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "demo/Transforms/Passes.h"
#include "demo/IR/DemoDialect.h"

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::demo::registerDemoPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::demo::DemoDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Demo optimizer driver\n", registry));
}
