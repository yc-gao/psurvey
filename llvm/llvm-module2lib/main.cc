#include "llvm/IR/LLVMContext.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/TargetSelect.h"

namespace {
llvm::cl::opt<std::string> InputFile(llvm::cl::Positional, llvm::cl::init("-"));
llvm::cl::opt<std::string> OutputFile(llvm::cl::Positional,
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
  auto Target = TargetRegistry::lookupTarget(TargetTriple, Error);
  if (!Target) {
    llvm::errs() << Error;
    return 1;
  }

  auto CPU = "generic";
  auto Features = "";
  TargetOptions opt;
  auto TheTargetMachine = Target->createTargetMachine(
      TargetTriple, CPU, Features, opt, Reloc::PIC_);
  m->setDataLayout(TheTargetMachine->createDataLayout());

  std::error_code EC;
  raw_fd_ostream dest(OutputFile, EC, sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    return 1;
  }

  legacy::PassManager pass;
  auto FileType = CodeGenFileType::ObjectFile;

  if (TheTargetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
    llvm::errs() << "TheTargetMachine can't emit a file of this type";
    return 1;
  }
  pass.run(*m);
  dest.flush();
  return 0;
}
