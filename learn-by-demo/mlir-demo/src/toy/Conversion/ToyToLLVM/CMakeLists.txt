add_mlir_conversion_library(MLIRToyToLLVM
    Pass.cpp
    # DEPENDS
    #     MLIRToyConversionPassIncGen
    LINK_LIBS PUBLIC
        MLIRToy
)
target_include_directories(MLIRToyToLLVM PUBLIC
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
    ${CMAKE_CURRENT_BINARY_DIR}
)

