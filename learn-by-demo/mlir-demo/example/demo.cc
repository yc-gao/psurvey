#include "Demo/Conversion/DemoToLLVM.h"
#include "Demo/IR/DemoOps.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"

namespace {

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input toy file>"),
                                         llvm::cl::init("-"),
                                         llvm::cl::value_desc("filename"));

mlir::OwningOpRef<mlir::ModuleOp> LoadMLIR(mlir::MLIRContext &context) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (auto ec = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << ec.message() << '\n';
    return nullptr;
  }

  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  auto module = mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return nullptr;
  }
  return module;
}

}  // namespace

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "demo compiler");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::demo::DemoDialect>();
  mlir::MLIRContext context(registry);

  auto module = LoadMLIR(context);
  if (!module) {
    return 1;
  }

  mlir::PassManager pm(module.get()->getName());
  mlir::demo::AddPassesDemoToLLVM(pm);
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "can't run pass on module";
    return 1;
  }
  module->dump();

  return 0;
}
