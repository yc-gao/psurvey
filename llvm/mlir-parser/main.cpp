#include <string>

#include "mlir/IR/Dialect.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
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

  auto module = LoadMLIR(context, InputFile);
  llvm::outs() << *module;
  return 0;
}
