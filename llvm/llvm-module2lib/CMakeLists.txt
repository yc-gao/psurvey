set(LLVM_LINK_COMPONENTS
    Support
    Core
    AllTargetsAsmParsers
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsInfos
    AsmPrinter
    IR
)

file(GLOB SRCS *.cc)
add_llvm_executable(llvm-module2lib ${SRCS})


