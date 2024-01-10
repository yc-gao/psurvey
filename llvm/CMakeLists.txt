cmake_minimum_required(VERSION 3.17)
project(llvm-quickstart)

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the specified C++ standard for all targets")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe" CACHE STRING "Flags used by the compiler")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline" CACHE STRING "Flags used by the compiler during Debug builds")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "Flags used by the compiler during Release builds")

find_package(MLIR REQUIRED)
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

add_subdirectory(jit)
add_subdirectory(ir2lib)
add_subdirectory(mlir-demo)

