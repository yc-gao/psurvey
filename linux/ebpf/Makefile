CLANG = clang
BPFTOOL = bpftool

APPS = example

all: $(APPS)

clean:
	rm -rf *.skel.h *.o $(APPS)

vmlinux.h:
	bpftool btf dump file /sys/kernel/btf/vmlinux format c > $@

example: example.cc example_bpf.skel.h
	$(CXX) -o $@ $< -lbpf -lelf -lz

example_bpf.skel.h: example.bpf.o
	$(BPFTOOL) gen skeleton $< > $@

example.bpf.o: example.bpf.tmp.o
	$(BPFTOOL) gen object $@ $^

example.bpf.tmp.o: example.bpf.c
	$(CLANG) -target bpf -o $@ -g -O2 -c $^
