all: demo

run: demo
	./demo

pack: demo
	./pack -L /usr/lib -L /usr/lib64 ./demo

.PHONY: all run pack

demo: demo.cc
