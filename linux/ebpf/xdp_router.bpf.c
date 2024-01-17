#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/in.h>
#include <bpf/bpf_helpers.h>

SEC("xdp_router")
int xdp_ip_router(struct xdp_md *ctx) {
  bpf_printk("xdp_ip_router");
  return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
