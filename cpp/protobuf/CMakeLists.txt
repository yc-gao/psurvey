cmake_minimum_required(VERSION 3.17)
project(demo)

include_directories(${CMAKE_CURRENT_BINARY_DIR})

find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS foo.proto)
add_library(proto ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(proto protobuf::libprotobuf)
target_include_directories(proto PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

add_executable(demo demo.cc)
target_link_libraries(demo proto)

