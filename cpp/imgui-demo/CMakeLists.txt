cmake_minimum_required(VERSION 3.17)
project(imgui-demo VERSION 0.0.1)

include(GNUInstallDirs)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
message(STATUS "setting C++ standard to C++${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -pipe")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -rdynamic -fno-omit-frame-pointer -fno-optimize-sibling-calls -fno-inline")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include;${CMAKE_CURRENT_SOURCE_DIR}/src")

add_subdirectory(3rdparty)

file(GLOB_RECURSE SRCS src/*.cc)
add_executable(demo ${SRCS})
target_link_libraries(demo PUBLIC imgui)

