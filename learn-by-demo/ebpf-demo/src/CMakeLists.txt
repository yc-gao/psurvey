find_package(bpf REQUIRED)

bpf_object(minimal_bpf minimal.bpf.c)

add_executable(minimal minimal.c)
target_link_libraries(minimal bpf::libbpf minimal_bpf elf z)

