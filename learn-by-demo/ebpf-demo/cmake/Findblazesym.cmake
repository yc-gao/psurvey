if(blazesym_ADDED)
    return()
endif()

CPMAddPackage("gh:libbpf/blazesym#v0.2.0-rc.0")
execute_process(
    WORKING_DIRECTORY ${blazesym_SOURCE_DIR}
    COMMAND cargo build --package=blazesym-c --release
)

find_path(blazesym_INCLUDE_DIR NAMES blazesym.h PATHS ${blazesym_SOURCE_DIR}/capi/include REQUIRED)
find_library(blazesym_LIBRARY NAMES libblazesym_c.a PATHS ${blazesym_SOURCE_DIR}/target/release REQUIRED)

add_library(blazesym::blazesym STATIC IMPORTED)
set_target_properties(blazesym::blazesym PROPERTIES
    IMPORTED_LOCATION "${blazesym_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${blazesym_INCLUDE_DIR}"
)
target_link_libraries(blazesym::blazesym INTERFACE pthread rt dl m)
add_dependencies(blazesym::blazesym blazesym-proj)

