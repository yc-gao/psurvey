add_executable(demo demo.cc)
target_compile_features(demo PRIVATE cxx_std_17)
target_link_libraries(demo PRIVATE fmt)

add_library(add add.cc)
target_include_directories(add PUBLIC
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)

install(TARGETS demo add
    EXPORT demoTargets
)
install(EXPORT demoTargets
    FILE demoConfig.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/demo
)

