cmake_minimum_required(VERSION 3.17)
project(gkernels VERSION 0.0.1 LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the specified C++ standard for all targets")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

set(CMAKE_CUDA_ARCHITECTURES 80 90)

find_package(CUDAToolkit REQUIRED)

add_executable(ptx_jit ptx_jit.cc)
target_link_libraries(ptx_jit PUBLIC CUDA::cudart CUDA::cuda_driver)

add_executable(kernel2ptx kernel2ptx.cc)
target_link_libraries(kernel2ptx PUBLIC CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)

add_executable(reduce reduce.cu)
target_link_libraries(reduce PUBLIC CUDA::cudart)

