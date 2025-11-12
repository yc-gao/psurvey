macro(add_benchmark name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE
        nvbench::nvbench
        nvbench::main
    )
endmacro()

add_benchmark(copy copy.cu)

add_benchmark(cute_copy cute_copy.cu)
target_link_libraries(cute_copy PRIVATE
    CUTLASS
)

