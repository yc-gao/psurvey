set(LLVM_LINK_COMPONENTS
    Support
    Core
    ExecutionEngine
    nativecodegen
    MC
    MCJIT
    Interpreter)

file(GLOB SRCS *.cc)
add_llvm_executable(jit ${SRCS})

