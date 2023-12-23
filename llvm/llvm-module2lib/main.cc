#include "llvm/IR/LLVMContext.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/TargetSelect.h"

int main(int argc, char *argv[]) {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  llvm::PassRegistry *Registry = llvm::PassRegistry::getPassRegistry();

  llvm::LLVMContext Context;
  return 0;
}
