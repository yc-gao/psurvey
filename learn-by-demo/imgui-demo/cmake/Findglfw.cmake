include(FetchContent)

set(GLFW_BUILD_WAYLAND OFF)
FetchContent_Declare(glfw URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz)
FetchContent_MakeAvailable(glfw)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(glfw DEFAULT_MSG
    glfw_POPULATED
)

target_link_libraries(glfw PUBLIC GL)
