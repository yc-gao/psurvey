find_package(imgui REQUIRED)

add_executable(demo demo.cc)
target_link_libraries(demo PRIVATE imgui)

