all: main preload.so test

clean:
	rm -rf main *.so *.o test

.PHONY: all clean


main: main.cpp stacktrace.cpp
	$(CXX) -o $@ -rdynamic $^ -ldl

preload.so: preload.cpp stacktrace.cpp
	$(CXX) -o $@ -fPIC -shared $^ -ldl

test: test.cpp

