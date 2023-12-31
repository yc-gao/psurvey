cmake_minimum_required(VERSION 3.17)
project(ebpf)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

find_package(bpf REQUIRED)

bpf_executable(example SRCS example.cc BPF_SRCS example.bpf.c)
bpf_executable(kprobe SRCS kprobe.cc BPF_SRCS kprobe.bpf.c)

