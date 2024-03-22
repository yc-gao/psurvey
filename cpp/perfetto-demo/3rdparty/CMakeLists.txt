include(FetchContent)

FetchContent_Declare(
    perfetto
    GIT_REPOSITORY      https://github.com/google/perfetto.git
    GIT_TAG             main
)
FetchContent_Populate(perfetto)

find_package(Protobuf REQUIRED)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${perfetto_SOURCE_DIR}/protos/perfetto/trace/perfetto_trace.proto)

add_library(perfetto_trace ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(perfetto_trace protobuf::libprotobuf)
target_include_directories(perfetto_trace PUBLIC ${CMAKE_CURRENT_BINARY_DIR})


