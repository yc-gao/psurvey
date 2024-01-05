file(GLOB SRCS *.cc)
add_executable(add ${SRCS})
target_link_libraries(add degine)

