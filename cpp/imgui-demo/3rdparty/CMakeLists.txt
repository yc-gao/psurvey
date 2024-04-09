include(FetchContent)

set(GLFW_BUILD_WAYLAND OFF)
FetchContent_Declare(glfw URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz)
FetchContent_MakeAvailable(glfw)

FetchContent_Declare(imgui
    GIT_REPOSITORY      https://github.com/ocornut/imgui.git
    GIT_TAG             docking
)
FetchContent_Populate(imgui)

add_library(imgui
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw dl GL)

