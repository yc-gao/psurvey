set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(MLIRDemoTransformsIncGen)

add_mlir_dialect_library(MLIRDemoTransforms
    DemoNormalizePass.cc
    DEPENDS
    MLIRDemoTransformsIncGen
)
