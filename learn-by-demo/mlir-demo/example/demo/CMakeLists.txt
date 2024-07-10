set(CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR};${CMAKE_MODULE_PATH}")
include(AddLLVM)
include(AddMLIR)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

add_llvm_executable(demo demo.cc)
target_include_directories(demo PRIVATE
    ${LLVM_INCLUDE_DIRS}
    ${MLIR_INCLUDE_DIRS}
)
target_link_libraries(demo PRIVATE
    LLVMSupport
    LLVMCore
    MLIRToLLVMIRTranslationRegistration
    MLIRExecutionEngine
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}
)
