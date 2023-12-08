all: demo

prof: demo
	./demo
	gprof ./demo gmon.out

demo: demo.cc
	$(CXX) -o $@ -pg $^
