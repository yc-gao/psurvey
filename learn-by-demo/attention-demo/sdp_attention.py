#!/usr/bin/env python3
import math

import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v):
    scale_factor = 1 / math.sqrt(q.size(-1))
    x = q @ k.transpose(-2, -1) * scale_factor
    x = torch.softmax(x, dim=-1)
    return x @ v


def main():
    q = torch.rand(256, 4096)
    k = torch.rand(128, 4096)
    v = torch.rand(128, 2048)
    pred = scaled_dot_product_attention(q, k, v)
    ref = F.scaled_dot_product_attention(q, k, v)
    print(torch.allclose(pred, ref))


if __name__ == "__main__":
    main()
