get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support)

file(GLOB SRCS *.cpp)
add_llvm_executable(mlir-parser ${SRCS})

target_link_libraries(mlir-parser
    PRIVATE
    ${dialect_libs}
    ${conversion_libs})

