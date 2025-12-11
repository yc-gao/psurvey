cmake_minimum_required(VERSION 3.17)
project(cutlass-demo
    VERSION 0.0.1
    LANGUAGES C CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_POSITION_INDEPENDENT_CODE True)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(proj_COMMON_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR};${CMAKE_CURRENT_BINARY_DIR}")

find_package(CUDAToolkit REQUIRED)
add_subdirectory(3rdparty)

add_executable(demo
    demo.cu
)
target_link_libraries(demo PRIVATE
    CUDA::cudart CUDA::cuda_driver
    nvidia::cutlass::cutlass
)

