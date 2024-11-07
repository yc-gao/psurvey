CPMAddPackage("gh:libbpf/bpftool#v7.4.0")

execute_process(
    WORKING_DIRECTORY ${bpftool_SOURCE_DIR}
    COMMAND         bash -c "make -C src \
                        && DESTDIR=${bpftool_BINARY_DIR} make -C src install"
    RESULT_VARIABLE bpftool_RESULT
)
if(bpftool_RESULT)
        message(FATAL_ERROR "Failed to build bpftool: ${bpftool_RESULT}")
endif()

execute_process(
    WORKING_DIRECTORY ${bpftool_SOURCE_DIR}/libbpf
    COMMAND         bash -c "make -C src BUILD_STATIC_ONLY=y \
                        && DESTDIR=${bpftool_BINARY_DIR} make -C src install \
                        && cp -rt ${bpftool_BINARY_DIR}/usr/include/ include/uapi/linux"
    RESULT_VARIABLE libbpf_RESULT
)
if(libbpf_RESULT)
        message(FATAL_ERROR "Failed to build libbpf: ${libbpf_RESULT}")
endif()

find_path(libbpf_INCLUDE_DIR NAMES bpf/bpf.h PATHS ${bpftool_BINARY_DIR}/usr/include REQUIRED)
find_library(libbpf_LIBRARY NAMES libbpf.a PATHS ${bpftool_BINARY_DIR}/usr/lib64 REQUIRED)
add_library(libbpf INTERFACE)
target_include_directories(libbpf INTERFACE "${libbpf_INCLUDE_DIR}")
target_link_libraries(libbpf INTERFACE "${libbpf_LIBRARY}")

find_program(BPFTOOL_EXE NAMES bpftool PATHS ${bpftool_BINARY_DIR}/usr/local/sbin/ REQUIRED)
find_program(CLANG_EXE NAMES clang REQUIRED)

function(bpf_object name)
    execute_process(
        COMMAND uname -m
        COMMAND sed -e "s/x86_64/x86/" -e "s/aarch64/arm64/" -e "s/ppc64le/powerpc/" -e "s/mips.*/mips/" -e "s/riscv64/riscv/"
        OUTPUT_VARIABLE ARCH_output
        ERROR_VARIABLE ARCH_error
        RESULT_VARIABLE ARCH_result
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(ARCH_result)
        message(FATAL_ERROR "Failed to determine target architecture: ${ARCH_error}")
    endif()
    set(ARCH ${ARCH_output})
    message(STATUS "BPF target arch: ${ARCH}")

    execute_process(
        COMMAND bash -c "${CLANG_EXE} -v -E - < /dev/null 2>&1"
        COMMAND sed -n "/<...> search starts here:/,/End of search list./p;"
        COMMAND sed "1d;$d;s/^ */ -I/;"
        OUTPUT_VARIABLE CLANG_SYSTEM_INCLUDES_output
        ERROR_VARIABLE CLANG_SYSTEM_INCLUDES_error
        RESULT_VARIABLE CLANG_SYSTEM_INCLUDES_result
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(CLANG_SYSTEM_INCLUDES_result)
      message(FATAL_ERROR "Failed to determine BPF system includes: ${CLANG_SYSTEM_INCLUDES_error}")
    endif()
    separate_arguments(CLANG_SYSTEM_INCLUDES UNIX_COMMAND ${CLANG_SYSTEM_INCLUDES_output})
    message(STATUS "BPF system include flags: ${CLANG_SYSTEM_INCLUDES}")

    set(${name}_skel ${CMAKE_CURRENT_BINARY_DIR}/${name}.skel.h)
    set(${name}_srcs "${ARGN}")
    list(TRANSFORM ${name}_srcs REPLACE .bpf.c .bpf.o OUTPUT_VARIABLE ${name}_objs)
    list(TRANSFORM ${name}_objs PREPEND ${CMAKE_CURRENT_BINARY_DIR}/ OUTPUT_VARIABLE ${name}_objs)
    list(TRANSFORM ${name}_srcs PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/ OUTPUT_VARIABLE ${name}_srcs)

    foreach(item IN ZIP_LISTS ${name}_objs ${name}_srcs)
        add_custom_command(
            OUTPUT ${item_0}
            COMMAND ${CLANG_EXE} --target=bpf -c -g -O2 -D__TARGET_ARCH_${ARCH}
                    ${CLANG_SYSTEM_INCLUDES} -I ${libbpf_INCLUDE_DIR} -I ${PROJECT_SOURCE_DIR}/include/vmlinux/${ARCH}
                    -o ${item_0} ${item_1}
            DEPENDS libbpf ${item_1}
        )
    endforeach()
    add_custom_command(
            OUTPUT ${${name}_skel}
            COMMAND ${BPFTOOL_EXE} gen object ${CMAKE_CURRENT_BINARY_DIR}/${name}.tmp.o ${${name}_objs}
            COMMAND ${BPFTOOL_EXE} gen skeleton ${CMAKE_CURRENT_BINARY_DIR}/${name}.tmp.o name ${name} > ${${name}_skel}
            VERBATIM
            DEPENDS ${${name}_objs}
    )
    add_library(${name} INTERFACE)
    target_sources(${name} INTERFACE ${${name}_skel})
    target_include_directories(${name} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${name} INTERFACE libbpf elf z)
endfunction()

function(bpf_executable name)
    cmake_parse_arguments(ARG "" "" "BPF_SRCS;SRCS;LINKS" ${ARGN})

    bpf_object(${name}_bpf ${ARG_BPF_SRCS})
    add_executable(${name} ${ARG_SRCS})
    target_link_libraries(${name} PRIVATE ${name}_bpf ${ARG_LINKS})
endfunction()

