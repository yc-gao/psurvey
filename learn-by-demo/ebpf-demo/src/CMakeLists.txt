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

bpf_executable(kprobe
    BPF_SRCS kprobe.bpf.c
    SRCS     kprobe.cc
    LINKS    blazesym
)
