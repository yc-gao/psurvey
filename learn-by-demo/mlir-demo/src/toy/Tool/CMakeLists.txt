get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(toy-opt demo.cc)
target_include_directories(toy-opt PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(toy-opt PRIVATE
    LLVMSupport
    LLVMCore
    MLIRToLLVMIRTranslationRegistration
    MLIRExecutionEngine
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)

