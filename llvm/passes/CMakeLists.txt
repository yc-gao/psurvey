# opt --load-pass-plugin=./build/pass-hello/hello.so -passes='hello' -S
add_llvm_pass_plugin(hello
    hello.cpp
)

