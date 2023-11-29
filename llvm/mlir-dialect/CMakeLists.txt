add_subdirectory(Demo)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_executable(mlir-dialect main.cc)
target_link_libraries(mlir-dialect PRIVATE ${dialect_libs} ${conversion_libs})

