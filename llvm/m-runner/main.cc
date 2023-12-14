#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Tools/ParseUtilities.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"

namespace {
llvm::cl::opt<std::string> InputFile(llvm::cl::Positional, llvm::cl::init("-"));
}

mlir::OwningOpRef<mlir::ModuleOp> LoadMLIR(mlir::MLIRContext *context,
                                           llvm::StringRef inputFilename) {
  std::string err;
  auto buf = mlir::openInputFile(inputFilename, &err);
  if (!buf) {
    llvm::errs() << err << '\n';
    return nullptr;
  }
  auto sourceMgr = std::make_shared<llvm::SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(buf), llvm::SMLoc());
  return mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, context);
}

std::unique_ptr<llvm::ExecutionEngine>
CreateEE(std::unique_ptr<llvm::Module> m) {
  std::string errStr;
  std::unique_ptr<llvm::ExecutionEngine> EE(
      llvm::EngineBuilder(std::move(m)).setErrorStr(&errStr).create());
  if (!EE) {
    llvm::errs() << errStr << '\n';
  }
  return EE;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "mlir runner");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::MLIRContext context(std::move(registry));

  auto m = LoadMLIR(&context, InputFile);
  llvm::outs() << "load mlir:\n";
  llvm::outs() << **m << '\n';

  mlir::PassManager pm(&context);
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  if (mlir::failed(pm.run(*m))) {
    return 1;
  }
  llvm::outs() << "lower to llvm:\n";
  llvm::outs() << **m << '\n';

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::LLVMContext llvmContext;
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*m, llvmContext, "main");
  llvm::outs() << "llvm ir:\n";

  std::unique_ptr<llvm::ExecutionEngine> EE = CreateEE(std::move(llvmModule));
  if (!EE) {
    return 1;
  }
  llvm::Function *main = EE->FindFunctionNamed("main");
  std::vector<llvm::GenericValue> Args;
  EE->runFunction(main, Args);
  return 0;
}
