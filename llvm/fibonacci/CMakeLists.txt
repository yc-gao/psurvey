set(LLVM_LINK_COMPONENTS
    Support
    Core
    ExecutionEngine
    nativecodegen
    MC
    MCJIT
    Interpreter)

file(GLOB SRCS *.cpp)
add_llvm_executable(fibonacci ${SRCS})

