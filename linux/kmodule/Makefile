EXTRA_CFLAGS = -Wall -g
obj-m += dummy.o
dummy-y = ops.o

KDIR = /lib/modules/`uname -r`/build

kbuild:
	make -C $(KDIR) M=`pwd` modules

clean:
	make -C $(KDIR) M=`pwd` clean
