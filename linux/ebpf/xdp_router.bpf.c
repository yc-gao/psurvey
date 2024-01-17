// load: sudo ip link set dev lo xdp obj build/xdp_router.bpf.o sec xdp
// ubload: sudo ip link set dev lo xdp off

#include <linux/bpf.h>

#include <bpf/bpf_helpers.h>

SEC("xdp")
int xdp_prog_simple(struct xdp_md *ctx) {
  return XDP_PASS;
}

char _license[] SEC("license") = "GPL";
