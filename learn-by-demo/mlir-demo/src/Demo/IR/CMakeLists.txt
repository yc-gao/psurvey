include_directories("${MLIR_INCLUDE_DIRS}")
add_mlir_dialect(DemoOps demo)

add_mlir_dialect_library(MLIRDemoDialect
    DemoOps.cc
    DEPENDS
        MLIRDemoOpsIncGen
)
target_include_directories(MLIRDemoDialect PUBLIC
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

