set(LLVM_LINK_COMPONENTS
    Support
    IRReader)

file(GLOB SRCS *.cpp)
add_llvm_executable(ir-parser ${SRCS})
