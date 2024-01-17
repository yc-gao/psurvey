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
set(libbpf_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/libbpf/libbpf.a")
add_library(bpf::bpf STATIC IMPORTED)
set_target_properties(bpf::bpf PROPERTIES
    IMPORTED_LOCATION "${libbpf_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES   "${libbpf_INCLUDE_DIRS}")
add_dependencies(bpf::bpf libbpf-proj)

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
set(bpftool_EXECUTABLE "${CMAKE_CURRENT_BINARY_DIR}/bpftoolbootstrap/bpftool")
add_executable(bpf::bpftool IMPORTED)
set_target_properties(bpf::bpftool PROPERTIES
    IMPORTED_LOCATION "${bpftool_EXECUTABLE}")
add_dependencies(bpf::bpftool bpftool-proj)

find_program(clang_EXECUTABLE NAMES clang-17 clang-16 clang-15 clang REQUIRED)

macro(bpf_generate_object objs)
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
    set(${objs})
    set(srcs "${ARGN}")
    string(REGEX REPLACE "\.c" ".o" tmp_objs "${srcs}")
    foreach(src obj IN ZIP_LISTS srcs tmp_objs)
        add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${obj}
            COMMAND ${clang_EXECUTABLE} -target bpf
                -o ${obj} -c -g -O2
                -D__TARGET_ARCH_${ARCH}
                -I${libbpf_INCLUDE_DIRS} -I/usr/include/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu
                ${CMAKE_CURRENT_SOURCE_DIR}/${src}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            DEPENDS bpf::bpf ${CMAKE_CURRENT_SOURCE_DIR}/${src})
        list(APPEND ${objs} ${CMAKE_CURRENT_BINARY_DIR}/${obj})
    endforeach()
endmacro()

macro(bpf_objects name)
  bpf_generate_object(${name}_objs ${ARGN})
  add_custom_target(${name} ALL DEPENDS ${${name}_objs})
endmacro()

macro(bpf_generate_skel skel)
    set(objs "${ARGN}")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${skel}.skel.h
        COMMAND bpf::bpftool gen object ${skel}.o ${objs}
        COMMAND bpf::bpftool gen skeleton ${skel}.o > ${skel}.skel.h
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        DEPENDS bpf::bpftool ${objs}
        VERBATIM)
endmacro()

macro(bpf_executable name)
    cmake_parse_arguments(ARG "" "" "SRCS;BPF_SRCS" ${ARGN})
    bpf_generate_object(objs ${ARG_BPF_SRCS})
    bpf_generate_skel(${name}_bpf ${objs})
    add_executable(${name} ${ARG_SRCS} ${CMAKE_CURRENT_BINARY_DIR}/${name}_bpf.skel.h)
    target_link_libraries(${name} bpf::bpf elf z)
    target_include_directories(${name} PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
endmacro()

