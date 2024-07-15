file(GLOB SRCS *.cu)
add_executable(main ${SRCS})
target_link_libraries(main PRIVATE
    nvbench::nvbench
    nvbench::main
)
