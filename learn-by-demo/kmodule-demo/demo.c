#include <linux/module.h>

MODULE_LICENSE("GPL");

int init_module(void) { return 0; }
void cleanup_module(void) {}
