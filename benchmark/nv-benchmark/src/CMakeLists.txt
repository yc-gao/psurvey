macro(add_benchmark name)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE
        nvbench::nvbench
        nvbench::main
        CUTLASS
    )
endmacro()

add_benchmark(cu_copy copy.cu)
add_benchmark(cu_gemm gemm.cu)

