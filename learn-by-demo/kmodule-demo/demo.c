#include <linux/init.h>
#include <linux/module.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Robert W. Oliver II");
MODULE_DESCRIPTION("A simple example Linux module.");
MODULE_VERSION("0.0.1");

static int __init demo_init(void) { return 0; }
static void __exit demo_uninit(void) {}

module_init(demo_init);
module_exit(demo_uninit);
