cmake_minimum_required(VERSION 3.17)
project(ebpf)

find_program(CLANG NAMES clang-17 clang-16 clang-15 clang REQUIRED)
find_program(BPFTOOL NAMES bpftool REQUIRED)

macro(bpf_object name)
    set(srcs ${ARGN})
    string(REGEX REPLACE "\.c" ".o" objs "${srcs}")

    foreach(src obj IN ZIP_LISTS srcs objs)
        add_custom_command(OUTPUT ${obj}
            COMMAND ${CLANG} -target bpf -o ${obj} -c -g -O2 ${CMAKE_CURRENT_SOURCE_DIR}/${src}
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

bpf_object(example_bpf example.bpf.c)

add_executable(example example.cc)
target_link_libraries(example example_bpf)

