get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(demo demo.cc)
llvm_update_compile_flags(demo)
target_include_directories(demo PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(demo PRIVATE
    MLIRToLLVMIRTranslationRegistration
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)
