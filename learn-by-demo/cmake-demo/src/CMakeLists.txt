add_executable(hello hello.cc)
add_library(add add.cc)
target_include_directories(add PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS hello add
    EXPORT demoTargets
)
install(EXPORT demoTargets
    FILE demoConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/demo
)

