#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

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

int main(int argc, char **argv) {
  llvm::cl::ParseCommandLineOptions(argc, argv, "llvm jiter");

  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();

  llvm::LLVMContext Context;
  llvm::SMDiagnostic Err;
  std::unique_ptr<llvm::Module> module =
      llvm::parseIRFile(InputFile, Err, Context);

  std::string errStr;
  std::unique_ptr<llvm::ExecutionEngine> EE(
      llvm::EngineBuilder(std::move(module)).setErrorStr(&errStr).create());

  if (!EE) {
    llvm::errs() << "Failed to construct ExecutionEngine: " << errStr << "\n";
    return 1;
  }

  // Call the Fibonacci function with argument n:
  llvm::Function *FibF = EE->FindFunctionNamed("fib");
  std::vector<llvm::GenericValue> Args(1);
  Args[0].IntVal = llvm::APInt(32, 10);
  llvm::GenericValue GV = EE->runFunction(FibF, Args);

  // import result of execution
  llvm::outs() << "Result: " << GV.IntVal << "\n";

  return 0;
}
