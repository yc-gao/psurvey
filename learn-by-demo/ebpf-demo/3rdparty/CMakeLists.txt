# curl --create-dirs -o cmake/CPM.cmake -fsSL https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM")
include(CPM)

add_subdirectory(blazesym)
add_subdirectory(bpf)

