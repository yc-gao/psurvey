# curl --create-dirs -o cmake/CPM.cmake -fsSL https://github.com/cpm-cmake/CPM.cmake/releases/latest/download/get_cpm.cmake
set(CPM_SOURCE_CACHE "$ENV{HOME}/.cache/CPM" CACHE STRING "cpm cache directory")
include(CPM)

CPMAddPackage("gh:NVIDIA/nvbench#main")
CPMAddPackage("gh:NVIDIA/cutlass#v3.5.1")

