KDIR ?= /lib/modules/`uname -r`/build

kbuild:
	$(MAKE) -C $(KDIR) M=$${PWD}

clean:
	$(MAKE) -C $(KDIR) M=$${PWD} clean

