set(LLVM_TARGET_DEFINITIONS DemoOps.td)
mlir_tablegen(DemoOps.h.inc -gen-op-decls)
mlir_tablegen(DemoOps.cpp.inc -gen-op-defs)
mlir_tablegen(DemoDialect.h.inc -gen-dialect-decls -dialect=demo)
mlir_tablegen(DemoDialect.cpp.inc -gen-dialect-defs -dialect=demo)
add_public_tablegen_target(MLIRDemoOpsIncGen)
add_dependencies(mlir-headers MLIRDemoOpsIncGen)

file(GLOB SRCS *.cpp)
add_mlir_dialect_library(MLIRDemo
    ${SRCS}
    DEPENDS
        MLIRDemoOpsIncGen
    LINK_LIBS PUBLIC
        MLIRIR)
target_include_directories(MLIRDemo PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR})
