find_package(MLIR REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR};${CMAKE_MODULE_PATH}")
include(TableGen)
include(AddLLVM)
include(AddMLIR)

