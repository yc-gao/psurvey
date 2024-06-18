file(GLOB SRCS *.cc *.cu *.cpp)

add_executable(cubench ${SRCS})
target_link_libraries(cubench PRIVATE nvbench::main CCCL::CCCL)

