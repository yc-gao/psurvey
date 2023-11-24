# opt --load-pass-plugin=./build/pass-hello/hello.so -passes='hello' -S
file(GLOB SRCS *.cpp)
add_llvm_pass_plugin(hello
    ${SRCS}
)

