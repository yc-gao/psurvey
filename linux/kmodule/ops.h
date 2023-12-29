#ifndef __OPS__
#define __OPS__

#include <linux/fs.h>

int dummy_open(struct inode *inode, struct file *file);
int dummy_release(struct inode *inode, struct file *file);

#endif
