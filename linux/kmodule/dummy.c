#include <linux/init.h>
#include <linux/module.h>
#include <linux/printk.h>

MODULE_LICENSE("GPL");

static __init int dummy_init(void) {
  pr_info("dummy init");
  return 0;
}

static __exit void dummy_exit(void) { pr_info("dummy exit"); }

module_init(dummy_init);
module_exit(dummy_exit);
