include(FetchContent)

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY  https://github.com/NVIDIA/nvbench.git
    GIT_TAG         a0f2fab72bc26dfe919aff326028c7f134ec7146
)
FetchContent_MakeAvailable(nvbench)

