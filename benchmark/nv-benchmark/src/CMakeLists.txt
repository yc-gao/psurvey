add_executable(demo demo.cu)
target_link_libraries(demo PRIVATE
    nvbench::nvbench
    nvbench::main
)
