#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/TargetParser/Host.h"

namespace {
llvm::cl::opt<std::string> InputFile(llvm::cl::Positional, llvm::cl::init("-"));
llvm::cl::opt<std::string> OutputFile("o", llvm::cl::init("-"),
                                      llvm::cl::init("output.o"));
} // namespace

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
  llvm::cl::ParseCommandLineOptions(argc, argv, "llvm as");

  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  llvm::LLVMContext Context;
  auto m = LoadIR(Context, InputFile);
  if (!m) {
    return 1;
  }

  auto TargetTriple = llvm::sys::getDefaultTargetTriple();
  m->setTargetTriple(TargetTriple);
  std::string Error;
  auto Target = llvm::TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    llvm::errs() << Error;
    return 1;
  }

  auto CPU = "generic";
  auto Features = "";
  llvm::TargetOptions opt;
  auto TheTargetMachine = Target->createTargetMachine(
      TargetTriple, CPU, Features, opt, llvm::Reloc::PIC_);
  m->setDataLayout(TheTargetMachine->createDataLayout());

  std::error_code EC;
  llvm::raw_fd_ostream dest(OutputFile, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    return 1;
  }

  llvm::legacy::PassManager pass;
  auto FileType = llvm::CodeGenFileType::CGFT_ObjectFile;

  if (TheTargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
    llvm::errs() << "TheTargetMachine can't emit a file of this type";
    return 1;
  }
  pass.run(*m);
  dest.flush();
  return 0;
}
