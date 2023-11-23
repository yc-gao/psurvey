set(LLVM_LINK_COMPONENTS
    Support
    BitWriter)

file(GLOB SRCS *.cpp)
add_llvm_executable(module-maker ${SRCS})

