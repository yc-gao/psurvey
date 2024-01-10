set(LLVM_LINK_COMPONENTS
    Support
    Core
    AllTargetsAsmParsers
    AllTargetsCodeGens
    AllTargetsDescs
    AllTargetsInfos
    AsmPrinter
)

file(GLOB SRCS *.cc)
add_llvm_executable(ir2lib ${SRCS})


