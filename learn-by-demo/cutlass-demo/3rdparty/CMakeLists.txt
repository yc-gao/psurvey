include(FetchContent)

set(CUTLASS_ENABLE_HEADERS_ONLY ON CACHE BOOL "Enable only the header library")
FetchContent_Declare(
    cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG v4.2.1
    EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(cutlass)

