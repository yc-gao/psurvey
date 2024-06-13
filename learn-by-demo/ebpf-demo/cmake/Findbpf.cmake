include(ExternalProject)

ExternalProject_Add(libbpf-proj
    GIT_REPOSITORY      https://github.com/libbpf/libbpf.git
    GIT_TAG             v1.4.0
    CONFIGURE_COMMAND   ""
    BUILD_IN_SOURCE     True
    BUILD_COMMAND       BUILD_STATIC_ONLY=y make -C src
    INSTALL_COMMAND     DESTDIR=<INSTALL_DIR> make -C src install
    COMMAND             cp -r include/uapi/linux <INSTALL_DIR>/usr/include/
)
ExternalProject_Get_Property(libbpf-proj INSTALL_DIR)

set(libbpf_prefix ${INSTALL_DIR})
set(libbpf_INCLUDE_DIRS ${libbpf_prefix}/usr/include)
set(libbpf_LIBRARIES "${libbpf_prefix}/usr/lib64/libbpf.a")
add_library(bpf::libbpf STATIC IMPORTED)
set_target_properties(bpf::libbpf PROPERTIES
    IMPORTED_LOCATION "${libbpf_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${libbpf_INCLUDE_DIRS}"
)
add_dependencies(bpf::libbpf libbpf-proj)


ExternalProject_Add(bpftool-proj
    GIT_REPOSITORY      https://github.com/libbpf/bpftool.git
    GIT_TAG             v7.4.0
    CONFIGURE_COMMAND   ""
    BUILD_IN_SOURCE     True
    BUILD_COMMAND       make -C src
    INSTALL_COMMAND     DESTDIR=<INSTALL_DIR> make -C src install
)
ExternalProject_Get_Property(bpftool-proj INSTALL_DIR)

set(bpftool_prefix ${INSTALL_DIR})
add_executable(bpf::bpftool IMPORTED)
set_target_properties(bpf::bpftool PROPERTIES
    IMPORTED_LOCATION "${bpftool_prefix}/usr/local/sbin/bpftool"
)
add_dependencies(bpf::bpftool bpftool-proj)


find_program(CLANG_EXE NAMES clang REQUIRED)


execute_process(
    COMMAND uname -m
    COMMAND sed -e "s/x86_64/x86/" -e "s/aarch64/arm64/" -e "s/ppc64le/powerpc/" -e "s/mips.*/mips/" -e "s/riscv64/riscv/"
    OUTPUT_VARIABLE ARCH_output
    ERROR_VARIABLE ARCH_error
    RESULT_VARIABLE ARCH_result
    OUTPUT_STRIP_TRAILING_WHITESPACE)
if(${ARCH_result} EQUAL 0)
    set(ARCH ${ARCH_output})
    message(STATUS "BPF target arch: ${ARCH}")
else()
    message(FATAL_ERROR "Failed to determine target architecture: ${ARCH_error}")
endif()

execute_process(
    COMMAND bash -c "${CLANG_EXE} -v -E - < /dev/null 2>&1"
    COMMAND sed -n "/<...> search starts here:/,/End of search list./p;"
    COMMAND sed "1d;$d;s/^ */ -I/;"
    OUTPUT_VARIABLE CLANG_SYSTEM_INCLUDES_output
    ERROR_VARIABLE CLANG_SYSTEM_INCLUDES_error
    RESULT_VARIABLE CLANG_SYSTEM_INCLUDES_result
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(${CLANG_SYSTEM_INCLUDES_result} EQUAL 0)
  separate_arguments(CLANG_SYSTEM_INCLUDES UNIX_COMMAND ${CLANG_SYSTEM_INCLUDES_output})
  message(STATUS "BPF system include flags: ${CLANG_SYSTEM_INCLUDES}")
else()
  message(FATAL_ERROR "Failed to determine BPF system includes: ${CLANG_SYSTEM_INCLUDES_error}")
endif()

macro(bpf_object name)
    set(${name}_skel ${CMAKE_CURRENT_BINARY_DIR}/${name}.skel.h)
    set(${name}_srcs "${ARGN}")
    list(TRANSFORM ${name}_srcs REPLACE .bpf.c .bpf.o OUTPUT_VARIABLE ${name}_objs)
    list(TRANSFORM ${name}_objs PREPEND ${CMAKE_CURRENT_BINARY_DIR}/ OUTPUT_VARIABLE ${name}_objs)
    list(TRANSFORM ${name}_srcs PREPEND ${CMAKE_CURRENT_SOURCE_DIR}/ OUTPUT_VARIABLE ${name}_srcs)

    foreach(item IN ZIP_LISTS ${name}_objs ${name}_srcs)
        add_custom_command(
            OUTPUT ${item_0}
            COMMAND ${CLANG_EXE} --target=bpf -c -g -O2 -D__TARGET_ARCH_${ARCH}
                    ${CLANG_SYSTEM_INCLUDES} -I ${libbpf_INCLUDE_DIRS} -I ${PROJECT_SOURCE_DIR}/vmlinux/${ARCH}
                    -o ${item_0} ${item_1}
            DEPENDS bpf::libbpf ${item_1}
        )
    endforeach()
    add_custom_command(
            OUTPUT ${${name}_skel}
            COMMAND bpf::bpftool gen object ${CMAKE_CURRENT_BINARY_DIR}/${name}.tmp.o ${${name}_objs}
            COMMAND bpf::bpftool gen skeleton ${CMAKE_CURRENT_BINARY_DIR}/${name}.tmp.o name ${name} > ${${name}_skel}
            VERBATIM
            DEPENDS bpf::bpftool ${${name}_objs}
    )
    add_library(${name} INTERFACE)
    target_sources(${name} INTERFACE ${${name}_skel})
    target_include_directories(${name} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${name} INTERFACE bpf::libbpf elf z)
endmacro()

