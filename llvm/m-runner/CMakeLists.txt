set(LLVM_LINK_COMPONENTS
    Support
    Core
    ExecutionEngine
    nativecodegen
    MC
    MCJIT
    Interpreter)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

file(GLOB SRCS *.cc)
add_llvm_executable(m-runner ${SRCS})
target_link_libraries(m-runner PRIVATE
    MLIRSupport
    MLIRIR
    MLIRParser
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs})

