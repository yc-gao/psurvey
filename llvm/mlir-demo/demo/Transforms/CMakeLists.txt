set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Demo)
add_public_tablegen_target(MLIRDemoPassIncGen)

get_property(dialect_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
file(GLOB SRCS *.cpp)
add_mlir_conversion_library(MLIRDemoToArith
    ${SRCS}
    LINK_LIBS PUBLIC
        ${dialect_libs})

target_include_directories(MLIRDemoToArith PUBLIC
    ${CMAKE_CURRENT_BINARY_DIR})
