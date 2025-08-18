include(FetchContent)

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY  https://github.com/NVIDIA/nvbench.git
    GIT_TAG         3de9dc95da3fc859900a799918a0c1d9091920ff
)
FetchContent_MakeAvailable(nvbench)

set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "Enable CUTLASS Examples")
set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "Enable CUTLASS Tools")
set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "Enable CUTLASS Tests")
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY  https://github.com/NVIDIA/cutlass.git
    GIT_TAG         main
)
FetchContent_MakeAvailable(cutlass)

