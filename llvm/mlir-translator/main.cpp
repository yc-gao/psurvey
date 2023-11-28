#include <string>

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllTranslations.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

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

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv);

  mlir::MLIRContext context;
  mlir::registerAllDialects(context);
  mlir::registerLLVMDialectTranslation(context);

  auto module = LoadMLIR(context, InputFile);
  mlir::PassManager pm(&context);
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "run passes failed";
    return 1;
  }

  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  llvm::outs() << *llvmModule;
  return 0;
}
