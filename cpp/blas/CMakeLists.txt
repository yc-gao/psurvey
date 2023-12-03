cmake_minimum_required(VERSION 3.17)
project(blas)

find_package(OpenBLAS REQUIRED)

add_executable(demo demo.cc)
target_link_libraries(demo OpenBLAS::OpenBLAS)

