find_package(bpf REQUIRED)

macro(bpf_executable name)
    cmake_parse_arguments(ARG "" "" "BPF_SRCS;SRCS" ${ARGN})

    bpf_object(${name}_bpf ${ARG_BPF_SRCS})
    add_executable(${name} ${ARG_SRCS})
    target_link_libraries(${name} PRIVATE ${name}_bpf)
endmacro()

bpf_executable(minimal
    BPF_SRCS minimal.bpf.c
    SRCS     minimal.c
)

bpf_executable(bootstrap
    BPF_SRCS bootstrap.bpf.c
    SRCS     bootstrap.cc
)
