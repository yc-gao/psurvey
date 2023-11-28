get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support)

file(GLOB SRCS *.cpp)
add_llvm_executable(mlir-passes ${SRCS})
llvm_update_compile_flags(mlir-passes)

target_link_libraries(mlir-passes
    PRIVATE
    ${dialect_libs}
    ${conversion_libs})

