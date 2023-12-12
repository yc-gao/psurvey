# cmake -B build -DCMAKE_PREFIX_PATH=/opt/miniconda3/lib/python3.8/site-packages/torch/ && cmake --build build

cmake_minimum_required(VERSION 3.17)
project(torch-script)

find_package(Torch REQUIRED)

add_executable(cpp_deploy main.cc)
target_link_libraries(cpp_deploy PRIVATE torch)
target_compile_features(cpp_deploy PRIVATE cxx_std_17)

