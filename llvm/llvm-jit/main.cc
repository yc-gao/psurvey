#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/GenericValue.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"

namespace {
llvm::cl::opt<std::string> InputFile(llvm::cl::Positional, llvm::cl::init("-"));
}

std::unique_ptr<llvm::Module> LoadIR(llvm::LLVMContext &Context,
                                     llvm::StringRef InputFile) {
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(InputFile, Err, Context);
  if (!module) {
    llvm::errs() << Err.getMessage() << '\n';
  }
  return module;
}

int main(int argc, char *argv[]) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "llvm jit");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::LLVMContext Context;
  auto m = LoadIR(Context, InputFile);
  if (!m) {
    return 1;
  }

  std::string errStr;
  std::unique_ptr<llvm::ExecutionEngine> EE =
      llvm::EngineBuilder(std::move(m)).setErrorStr(&errStr).create();
  if (!EE) {
    llvm::errs() << errStr << '\n';
    return 1;
  }
  void (*func)() = (void (*)())EE->getFunctionAddress("main");
  func();
  return 0;
}
