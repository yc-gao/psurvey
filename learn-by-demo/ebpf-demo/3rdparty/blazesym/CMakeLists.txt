CPMAddPackage("gh:libbpf/blazesym#v0.2.0-rc.0")
execute_process(
    WORKING_DIRECTORY ${blazesym_SOURCE_DIR}
    COMMAND cargo build --package=blazesym-c --release
)

find_path(blazesym_INCLUDE_DIR NAMES blazesym.h PATHS ${blazesym_SOURCE_DIR}/capi/include REQUIRED)
find_library(blazesym_LIBRARY NAMES libblazesym_c.a PATHS ${blazesym_SOURCE_DIR}/target/release REQUIRED)

add_library(blazesym INTERFACE)
target_include_directories(blazesym INTERFACE ${blazesym_INCLUDE_DIR})
target_link_libraries(blazesym INTERFACE ${blazesym_LIBRARY} pthread rt dl m)

