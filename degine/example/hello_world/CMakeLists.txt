find_package(gflags REQUIRED)

add_executable(hello_world main.cc)
target_link_libraries(hello_world PUBLIC degine gflags)

