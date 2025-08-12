include(FetchContent)

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY  https://github.com/NVIDIA/nvbench.git
    GIT_TAG         3de9dc95da3fc859900a799918a0c1d9091920ff
)
FetchContent_MakeAvailable(nvbench)

set(CUTLASS_ENABLE_HEADERS_ONLY ON)
set(CUTLASS_ENABLE_EXAMPLES OFF)
set(CUTLASS_ENABLE_TOOLS OFF)
set(CUTLASS_ENABLE_TESTS OFF)
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY  https://github.com/NVIDIA/cutlass.git
    GIT_TAG         main
)
FetchContent_MakeAvailable(cutlass)

