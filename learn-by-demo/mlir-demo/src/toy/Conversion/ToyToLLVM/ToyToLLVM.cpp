#include "ToyToLLVM.h"

#include "ToyConstantOpLoweringPass.h"
#include "ToyToLLVMLoweringPass.h"

namespace mlir {
namespace toy {

void addPassesToLLVM(mlir::PassManager &pm) {
  pm.addPass(createToyConstantOpLoweringPass());
  pm.addPass(createToyToLLVMLoweringPass());
}

}  // namespace toy
}  // namespace mlir
