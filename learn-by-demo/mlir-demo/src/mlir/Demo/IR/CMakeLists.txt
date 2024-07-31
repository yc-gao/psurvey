add_mlir_dialect(DemoOps demo)

add_mlir_dialect_library(MLIRDemoDialect
    DemoOps.cpp
    DEPENDS
        MLIRDemoOpsIncGen
)
