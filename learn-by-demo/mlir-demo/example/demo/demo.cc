#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ConvertToLLVM/ToLLVMPass.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

namespace {

llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                         llvm::cl::desc("<input toy file>"),
                                         llvm::cl::init("-"),
                                         llvm::cl::value_desc("filename"));

enum Action {
  DumpMLIR,
  DumpLLVMIR,
  DumpLLVM,
  JIT,
};
llvm::cl::opt<Action> emitAction(
    "emit",
    llvm::cl::values(clEnumValN(Action::DumpMLIR, "mlir", "output the mlir")),
    llvm::cl::values(clEnumValN(Action::DumpLLVMIR, "llvmir",
                                "output the llvmir")),
    llvm::cl::values(clEnumValN(Action::DumpLLVM, "llvm", "output the llvm")),
    llvm::cl::values(clEnumValN(Action::JIT, "jit", "llvm jit")));

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
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(argc, argv, "demo compiler");

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllExtensions(registry);
  mlir::registerAllToLLVMIRTranslations(registry);
  mlir::registerAllGPUToLLVMIRTranslations(registry);
  mlir::MLIRContext context(registry);

  context.loadAllAvailableDialects();
  auto module = LoadMLIR(context);
  if (!module) {
    return 1;
  }
  if (emitAction == Action::DumpMLIR) {
    module->dump();
    return 0;
  }

  mlir::PassManager pm(module.get()->getName());
  if (mlir::failed(mlir::applyPassManagerCLOptions(pm))) {
    llvm::errs() << "Error can't applyPassManagerCLOptions";
    return 1;
  }
  pm.addPass(mlir::tosa::createTosaToArith());
  pm.addPass(mlir::arith::createConstantBufferizePass(64));
  pm.addPass(mlir::createConvertToLLVMPass());
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "Error can't run PassManager";
    return 1;
  }
  if (emitAction == Action::DumpLLVMIR) {
    module->dump();
  }

  if (emitAction == Action::DumpLLVM) {
    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
    if (!llvmModule) {
      llvm::errs() << "Failed to emit LLVM IR\n";
      return 1;
    }
    llvm::errs() << *llvmModule;
    return 0;
  }

  if (emitAction == Action::JIT) {
    // llvm jit
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    mlir::ExecutionEngineOptions engineOptions;
    engineOptions.transformer = mlir::makeOptimizingTransformer(0, 0, nullptr);
    auto maybeEngine = mlir::ExecutionEngine::create(*module, engineOptions);
    if (!maybeEngine) {
      llvm::errs() << "Failed to create ExecutionEngine\n";
      return 1;
    }
    auto invocationResult = (*maybeEngine)->invokePacked("main");
    if (invocationResult) {
      llvm::errs() << "Failed to invokePacked\n";
      return 1;
    }
  }

  return 0;
}
