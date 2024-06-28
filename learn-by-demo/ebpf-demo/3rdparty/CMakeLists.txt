# curl --create-dirs -o cmake/CPM.cmake -fsSL https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM")
include(CPM)

CPMAddPackage("gh:microsoft/GSL#v4.0.0")
add_subdirectory(blazesym)
add_subdirectory(bpf)

