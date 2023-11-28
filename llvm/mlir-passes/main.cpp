#include <memory>
#include <string>

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/NVGPUToNVVM/NVGPUToNVVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"

namespace {
llvm::cl::opt<std::string> InputFile(llvm::cl::Positional, llvm::cl::init("-"));
}

mlir::OwningOpRef<mlir::ModuleOp> LoadMLIR(mlir::MLIRContext &context,
                                           std::string InputFile) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(InputFile);
  if (std::error_code ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << "\n";
    return nullptr;
  }
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
}

class PrintOperatorPass
    : public mlir::PassWrapper<PrintOperatorPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
public:
  void runOnOperation() override {
    // define Conversiion Tagrte
    // define Rewrite Patterns
    // maybe a Type Convertor
    // Convert Target
    auto op = getOperation();
    llvm::outs() << "PrintOperatorPass\n";
    llvm::outs() << *op;
    markAllAnalysesPreserved();
  }
};

std::unique_ptr<PrintOperatorPass> createPrintOperatorPass() {
  return std::make_unique<PrintOperatorPass>();
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::MLIRContext context(registry);

  auto module = LoadMLIR(context, InputFile);
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createConvertNVGPUToNVVMPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::createConvertVectorToLLVMPass());
  pm.addPass(createPrintOperatorPass());
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "run pm failed";
    return 1;
  }
  return 0;
}
