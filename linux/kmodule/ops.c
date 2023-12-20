#include "ops.h"

#include <linux/module.h>
#include <linux/printk.h>

MODULE_LICENSE("GPL");

int dummy_open(struct inode *inode, struct file *file) {
  pr_info("dummy open");
  return 0;
}

int dummy_release(struct inode *inode, struct file *file) {
  pr_info("dummy release");
  return 0;
}

struct file_operations dummy_ops = {
    .open = dummy_open,
    .release = dummy_release,
};
