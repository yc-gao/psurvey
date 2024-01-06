find_package(Threads REQUIRED)

include(FetchContent)
FetchContent_Declare(
    asio
    GIT_REPOSITORY      https://github.com/chriskohlhoff/asio.git
    GIT_TAG             asio-1-29-0
)
FetchContent_MakeAvailable(asio)
set(asio_INCLUDE_DIRS "${asio_SOURCE_DIR}/asio/include")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(asio DEFAULT_MSG asio_INCLUDE_DIRS Threads_FOUND)

add_library(asio::asio INTERFACE IMPORTED)
target_include_directories(asio::asio INTERFACE "${asio_SOURCE_DIR}/asio/include")
target_link_libraries(asio::asio INTERFACE Threads::Threads)

