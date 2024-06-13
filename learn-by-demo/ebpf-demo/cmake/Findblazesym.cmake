include(ExternalProject)

ExternalProject_Add(blazesym-proj
    GIT_REPOSITORY      https://github.com/libbpf/blazesym.git
    GIT_TAG             v0.2.0-rc.0
    CONFIGURE_COMMAND   ""
    BUILD_IN_SOURCE     True
    BUILD_COMMAND       cargo build --package=blazesym-c --release
    INSTALL_COMMAND     ""
)
ExternalProject_Get_Property(blazesym-proj SOURCE_DIR)

set(blazesym_prefix ${SOURCE_DIR})
set(blazesym_INCLUDE_DIRS "${blazesym_prefix}/capi/include")
set(blazesym_LIBRARIES "${blazesym_prefix}/target/release/libblazesym_c.a")
add_library(blazesym::blazesym STATIC IMPORTED)
set_target_properties(blazesym::blazesym PROPERTIES
    IMPORTED_LOCATION "${blazesym_LIBRARIES}"
    INTERFACE_INCLUDE_DIRECTORIES "${blazesym_INCLUDE_DIRS}"
)
target_link_libraries(blazesym::blazesym INTERFACE pthread rt dl m)
add_dependencies(blazesym::blazesym blazesym-proj)

