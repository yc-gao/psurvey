add_mlir_conversion_library(MLIRToyToLLVM
    Pass.cpp
    LINK_LIBS PUBLIC
        MLIRToy
)

