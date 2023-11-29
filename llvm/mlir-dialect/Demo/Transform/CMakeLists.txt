get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_conversion_library(MLIRDemoTransforms
    LowerToAffineLoops.cpp
    LINK_LIBS
        ${dialect_libs})
target_include_directories(MLIRDemoTransforms PUBLIC ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR})

