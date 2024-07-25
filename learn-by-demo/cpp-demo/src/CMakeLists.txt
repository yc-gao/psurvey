add_executable(doctest_demo doctest_demo.cc)
target_compile_features(doctest_demo PRIVATE cxx_std_17)
target_link_libraries(doctest_demo PRIVATE doctest::doctest)

