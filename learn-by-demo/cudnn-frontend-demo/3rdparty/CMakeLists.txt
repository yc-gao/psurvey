include(FetchContent)

set(CUDNN_FRONTEND_BUILD_SAMPLES OFF)
set(CUDNN_FRONTEND_BUILD_TESTS OFF)
set(CUDNN_FRONTEND_BUILD_PYTHON_BINDINGS OFF)
FetchContent_Declare(
    cudnn_frontend
    GIT_REPOSITORY  https://github.com/NVIDIA/cudnn-frontend.git
    GIT_TAG         v1.14.1
)
FetchContent_MakeAvailable(cudnn_frontend)

set(CCCL_ENABLE_THRUST ON)
set(THRUST_ENABLE_HEADER_TESTING OFF)
set(THRUST_ENABLE_TESTING OFF)
set(THRUST_ENABLE_EXAMPLES OFF)
FetchContent_Declare(
    cccl
    GIT_REPOSITORY  https://github.com/NVIDIA/cccl.git
    GIT_TAG         v3.0.3
)
FetchContent_MakeAvailable(cccl)

