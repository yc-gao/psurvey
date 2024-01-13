ifneq ($(KERNELRELEASE),)
obj-m += dummy.o
dummy-y := ops.o proc.o
else
KDIR = /lib/modules/`uname -r`/build

default:
	make -C $(KDIR) M=`pwd`

clean:
	make -C $(KDIR) M=`pwd` clean
endif
