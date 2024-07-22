cmake_minimum_required(VERSION 3.17)
project(pybind11-demo
    VERSION 0.0.1
    LANGUAGES C CXX)

set(CMAKE_MODULE_PATH "${CMAKE_MODULE_PATH};${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-pipe -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_POSITION_INDEPENDENT_CODE True)
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include")

include(GNUInstallDirs)

add_subdirectory(3rdparty)

pybind11_add_module(example example.cc)
install(TARGETS example LIBRARY DESTINATION .)

