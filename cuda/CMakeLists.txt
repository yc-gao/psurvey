cmake_minimum_required(VERSION 3.17)
project(gkernels LANGUAGES CXX CUDA)

# cxx config
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require the specified C++ standard for all targets")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe" CACHE STRING "Flags used by the compiler")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline" CACHE STRING "Flags used by the compiler during Debug builds")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3" CACHE STRING "Flags used by the compiler during Release builds")

find_package(CUDAToolkit REQUIRED)

add_executable(ptx_jit ptx_jit.cc)
target_link_libraries(ptx_jit PUBLIC CUDA::cudart CUDA::cuda_driver)

add_executable(kernel2ptx kernel2ptx.cc)
target_link_libraries(kernel2ptx PUBLIC CUDA::cudart CUDA::nvrtc CUDA::cuda_driver)

