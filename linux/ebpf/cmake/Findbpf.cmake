include(ExternalProject)
ExternalProject_Add(libbpf-proj
    PREFIX libbpf
    GIT_REPOSITORY  https://github.com/libbpf/libbpf.git
    GIT_TAG         20c0a9e3d7e7d4aeb283eae982543c9cacc29477
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make
        -C src
        BUILD_STATIC_ONLY=1
        DESTDIR=${CMAKE_CURRENT_BINARY_DIR}/libbpf
        INCLUDEDIR=
        LIBDIR=
        UAPIDIR=
        install install_uapi_headers
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND ""
    STEP_TARGETS build
)
set(libbpf_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/libbpf")
add_library(bpf STATIC IMPORTED)
set_target_properties(bpf PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a"
    INTERFACE_INCLUDE_DIRECTORIES   "${libbpf_INCLUDE_DIRS}")

ExternalProject_Add(bpftool-proj
    PREFIX bpftool
    GIT_REPOSITORY  https://github.com/libbpf/bpftool.git
    GIT_TAG         v7.3.0
    CONFIGURE_COMMAND ""
    BUILD_COMMAND make bootstrap
        -C src
        OUTPUT=${CMAKE_CURRENT_BINARY_DIR}/bpftool
    BUILD_IN_SOURCE TRUE
    INSTALL_COMMAND ""
    STEP_TARGETS build
)
add_executable(bpftool IMPORTED)
set_target_properties(bpftool PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/bpftoolbootstrap/bpftool")

find_program(CLANG NAMES clang-17 clang-16 clang-15 clang REQUIRED)

macro(bpf_object name)
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

    set(srcs ${ARGN})
    string(REGEX REPLACE "\.c" ".o" objs "${srcs}")

    foreach(src obj IN ZIP_LISTS srcs objs)
        add_custom_command(OUTPUT ${obj}
            COMMAND ${CLANG} -target bpf
                -o ${obj} -c -g -O2
                -D__TARGET_ARCH_${ARCH}
                -I${libbpf_INCLUDE_DIRS} -I/usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                ${CMAKE_CURRENT_SOURCE_DIR}/${src}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${src})
    endforeach()
    add_custom_command(OUTPUT ${name}.skel.h
        COMMAND bpftool gen object ${name}.o ${objs}
        COMMAND bpftool gen skeleton ${name}.o > ${name}.skel.h
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


