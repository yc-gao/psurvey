#include <iostream>

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/IR/Module.h"

namespace
{
    llvm::cl::opt<std::string> InputFile(llvm::cl::Positional, llvm::cl::init("-"));
}

int main(int argc, char const *argv[])
{
    llvm::cl::ParseCommandLineOptions(argc, argv, "llvm ir reader");

    llvm::LLVMContext Context;
    llvm::SMDiagnostic Err;
    auto module = llvm::parseIRFile(InputFile, Err, Context);
    module->dump();
    return 0;
}
