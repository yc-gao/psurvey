get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_conversion_library(MLIRToyToLLVM
    ToyToLLVM.cpp
    ToyConstantOpLoweringPass.cpp
    ToyToLLVMLoweringPass.cpp
    LINK_LIBS PUBLIC
        ${dialect_libs}
)
target_include_directories(MLIRToyToLLVM PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

