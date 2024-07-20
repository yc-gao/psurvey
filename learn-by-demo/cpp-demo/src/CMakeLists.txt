file(GLOB SRCS *.cc)
add_executable(demo ${SRCS})
target_compile_features(demo PRIVATE cxx_std_17)
target_link_libraries(demo PRIVATE doctest::doctest)

