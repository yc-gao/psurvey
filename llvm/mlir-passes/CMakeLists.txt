get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)

set(LLVM_LINK_COMPONENTS
  Core
  Support)

file(GLOB SRCS *.cpp)
add_llvm_executable(mlir-passes ${SRCS})

target_link_libraries(mlir-passes
    PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    ${translation_libs})

