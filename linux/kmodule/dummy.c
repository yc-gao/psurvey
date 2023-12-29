#include <linux/kernel.h>
#include <linux/module.h>
#include <linux/printk.h>

#include "proc.h"

static __init int dummy_init(void) {
  int err;
  err = proc_init();
  if (err) {
    pr_err("proc init failed, code: %d", err);
    return 1;
  }
  pr_info("dummy init");
  return 0;
}

static __exit void dummy_exit(void) {
  proc_exit();
  pr_info("dummy exit");
}

module_init(dummy_init);
module_exit(dummy_exit);

MODULE_LICENSE("GPL");
