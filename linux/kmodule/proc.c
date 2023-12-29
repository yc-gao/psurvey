#include "proc.h"

#include <linux/err.h>
#include <linux/module.h>
#include <linux/proc_fs.h>

#include "ops.h"

struct proc_ops ops = {
    .proc_open = dummy_open,
    .proc_release = dummy_release,
};
struct proc_dir_entry *entry = NULL;

int proc_init() {
  entry = proc_create("demo", 0666, NULL, &ops);
  if (entry == NULL) {
    return -ENOMEM;
  }
  return 0;
}
void proc_exit() { remove_proc_entry("demo", NULL); }

MODULE_LICENSE("GPL");
