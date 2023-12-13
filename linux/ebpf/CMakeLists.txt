cmake_minimum_required(VERSION 3.17)
project(ebpf)

find_program(CLANG NAMES clang-17 clang-16 clang-15 clang REQUIRED)
find_program(BPFTOOL NAMES bpftool REQUIRED)

if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64")
  set(ARCH "x86")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm")
  set(ARCH "arm")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
  set(ARCH "arm64")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "ppc64le")
  set(ARCH "powerpc")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "mips")
  set(ARCH "mips")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "riscv64")
  set(ARCH "riscv")
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "loongarch64")
  set(ARCH "loongarch")
endif()
macro(bpf_object name)
    set(srcs ${ARGN})
    string(REGEX REPLACE "\.c" ".o" objs "${srcs}")

    foreach(src obj IN ZIP_LISTS srcs objs)
        add_custom_command(OUTPUT ${obj}
            COMMAND ${CLANG} -target bpf -o ${obj} -c -D__TARGET_ARCH_${ARCH} -g -O2 ${CMAKE_CURRENT_SOURCE_DIR}/${src}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    endforeach()
    add_custom_command(OUTPUT ${name}.skel.h
        COMMAND ${BPFTOOL} gen object ${name}.o ${objs}
        COMMAND ${BPFTOOL} gen skeleton ${name}.o > ${name}.skel.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS ${objs}
        VERBATIM)
    add_library(${name} INTERFACE ${name}.skel.h)
    target_include_directories(${name} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${name} INTERFACE bpf elf z)
endmacro()

macro(bpf_executable name)
    cmake_parse_arguments(ARG "" "" "SRCS;BPF_SRCS" ${ARGN})
    bpf_object(${name}_bpf ${ARG_BPF_SRCS})
    add_executable(${name} ${ARG_SRCS})
    target_link_libraries(${name} ${name}_bpf)
endmacro()

bpf_executable(example SRCS example.cc BPF_SRCS example.bpf.c)
bpf_executable(kprobe SRCS kprobe.cc BPF_SRCS kprobe.bpf.c)


