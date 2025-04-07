macro(add_benchmark name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE
        nvbench::nvbench
        nvbench::main
    )
endmacro()

add_benchmark(copy copy.cu)

