get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(demo main.cc)
llvm_update_compile_flags(demo)
target_link_libraries(demo PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)
