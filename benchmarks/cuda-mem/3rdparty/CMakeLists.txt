include(FetchContent)

FetchContent_Declare(nvbench
    GIT_REPOSITORY  https://github.com/NVIDIA/nvbench.git
    GIT_TAG         main
)
FetchContent_MakeAvailable(nvbench)

