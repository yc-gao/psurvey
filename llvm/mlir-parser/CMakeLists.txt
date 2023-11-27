set(LLVM_LINK_COMPONENTS
  Core
  Support)

file(GLOB SRCS *.cpp)
add_llvm_executable(mlir-parser ${SRCS})

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
target_link_libraries(mlir-parser
    PRIVATE
    ${dialect_libs})

