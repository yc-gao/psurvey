find_package(bpf REQUIRED)
find_package(blazesym REQUIRED)

macro(bpf_executable name)
    cmake_parse_arguments(ARG "" "" "BPF_SRCS;SRCS;LINKS" ${ARGN})

    bpf_object(${name}_bpf ${ARG_BPF_SRCS})
    add_executable(${name} ${ARG_SRCS})
    target_link_libraries(${name} PRIVATE ${name}_bpf ${ARG_LINKS})
endmacro()

bpf_executable(minimal
    BPF_SRCS minimal.bpf.c
    SRCS     minimal.c
)

bpf_executable(execsnoop
    BPF_SRCS execsnoop.bpf.c
    SRCS     execsnoop.cc
)

bpf_executable(uprobe
    BPF_SRCS uprobe.bpf.c
    SRCS     uprobe.cc
    LINKS    blazesym::blazesym
)
