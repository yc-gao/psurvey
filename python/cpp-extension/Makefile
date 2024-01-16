all: libadd.so

clean:
	rm -rf *.so

lib%.so: %.c
	$(CC) -o $@ -fPIC -shared $^
