include_directories(${MLIR_INCLUDE_DIRS})
add_mlir_dialect(ToyOps toy)

add_mlir_dialect_library(MLIRToy
    ToyOps.cpp
    DEPENDS
        MLIRToyOpsIncGen
)
target_include_directories(MLIRToy PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

