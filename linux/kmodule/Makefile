obj-m += dummy.o
dummy-y = ops.o proc.o

KDIR = /lib/modules/`uname -r`/build

kbuild:
	make -C $(KDIR) M=`pwd` modules

clean:
	make -C $(KDIR) M=`pwd` clean
