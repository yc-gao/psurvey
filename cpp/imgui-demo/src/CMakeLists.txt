find_package(imgui REQUIRED)

add_executable(demo main.cc)
target_link_libraries(demo PUBLIC imgui)

