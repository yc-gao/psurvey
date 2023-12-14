cmake_minimum_required(VERSION 3.17)
project(cu-survey)

enable_language(CUDA)
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

add_executable(gemm_cublas gemm_cublas.cu)
target_link_libraries(gemm_cublas cublas)
