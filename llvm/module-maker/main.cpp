#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/Support/raw_ostream.h"

int main() {
  llvm::LLVMContext Context;

  // Create the "module" or "program" or "translation unit" to hold the
  // function
  std::unique_ptr<llvm::Module> M =
      std::make_unique<llvm::Module>("test", Context);

  // Create the main function: first create the type 'int ()'
  llvm::FunctionType *FT = llvm::FunctionType::get(
      llvm::Type::getInt32Ty(Context), /*not vararg*/ false);

  // By passing a module as the last parameter to the Function constructor,
  // it automatically gets appended to the Module.
  llvm::Function *F = llvm::Function::Create(
      FT, llvm::Function::ExternalLinkage, "main", M.get());

  // Add a basic block to the function... again, it automatically inserts
  // because of the last argument.
  llvm::BasicBlock *BB = llvm::BasicBlock::Create(Context, "EntryBlock", F);

  // Get pointers to the constant integers...
  llvm::Value *Two = llvm::ConstantInt::get(llvm::Type::getInt32Ty(Context), 2);
  llvm::Value *Three =
      llvm::ConstantInt::get(llvm ::Type::getInt32Ty(Context), 3);

  // Create the add instruction... does not insert...
  llvm::Instruction *Add = llvm::BinaryOperator::Create(
      llvm::Instruction::Add, Two, Three, "addresult");

  // explicitly insert it into the basic block...
  Add->insertInto(BB, BB->end());

  // Create the return instruction and add it to the basic block
  llvm::ReturnInst::Create(Context, Add)->insertInto(BB, BB->end());

  // Output the bitcode file to stdout
  llvm::WriteBitcodeToFile(*M, llvm::outs());

  return 0;
}
