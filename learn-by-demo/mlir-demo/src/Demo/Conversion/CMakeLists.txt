get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_conversion_library(MLIRDemoToLLVM
    DemoToLLVM.cpp
    LINK_LIBS PUBLIC
        ${dialect_libs}
)

