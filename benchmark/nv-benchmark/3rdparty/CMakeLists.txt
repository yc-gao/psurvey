include(FetchContent)

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY  https://github.com/NVIDIA/nvbench.git
    GIT_TAG         3de9dc95da3fc859900a799918a0c1d9091920ff
)
FetchContent_MakeAvailable(nvbench)


