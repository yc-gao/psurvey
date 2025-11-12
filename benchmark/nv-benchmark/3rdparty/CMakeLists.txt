include(FetchContent)

FetchContent_Declare(
    nvbench
    GIT_REPOSITORY  https://github.com/NVIDIA/nvbench.git
    GIT_TAG         3de9dc95da3fc859900a799918a0c1d9091920ff
)
FetchContent_MakeAvailable(nvbench)

include(ExternalProject)

ExternalProject_Add(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v4.2.1
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)

ExternalProject_Get_property(cutlass SOURCE_DIR)
add_library(CUTLASS INTERFACE)
target_include_directories(CUTLASS INTERFACE
    "${SOURCE_DIR}/include"
)

