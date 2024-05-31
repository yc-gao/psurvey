#!/usr/bin/env python3
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: impl
# def multihead_attention(q, k, v, w_q, w_k, w_v, w_o):
#     F.scaled_dot_product_attention(q @ w_q, k @ w_k, v @ w_v)
#     pass

def main():
    q = torch.rand(256, 1024)
    multihead_attn = nn.MultiheadAttention(q.size(-1), 4)
    ref, _ = multihead_attn(q, q, q)
    print(ref.shape)


if __name__ == "__main__":
    main()
