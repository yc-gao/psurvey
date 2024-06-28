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
    LINKS    blazesym
)
target_link_libraries(uprobe PRIVATE GSL)
