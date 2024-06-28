"""
attention.py only checks attention is correct given the QKV input in shape of (B, T, 3*C)

Here we want to create some test data to check that the full attention layer,
going from shape (B, C, H, W) to (B, C, H, W), is correct.
"""
import torch
import torch.nn as nn
from utils import write
from unet import AttentionBlock

if __name__ == "__main__":
    B = 32
    C = 64
    H = 32
    W = 32
    HS = 8
    NH = C // HS
    gn_n_groups = 32
    
    attention = AttentionBlock(C, NH, HS, gn_n_groups, debug=True)
    
    x = torch.randn(B, C, H, W, requires_grad=True)
    out, gn, qkv, att, proj = attention(x)

    # we permute the intermediate states, so also permute them for comparison
    perm1 = gn.permute(0, 2, 1).contiguous()
    qkv = qkv.permute(0, 2, 1).contiguous()
    att = att.permute(0, 2, 1).contiguous()
    proj = proj.permute(0, 2, 1).contiguous()
    print(f"shapes: x={x.shape}, out={out.shape}, gn={gn.shape}, qkv={qkv.shape}, att={att.shape}, proj={proj.shape}")
    total_states = x.numel() + gn.numel() + qkv.numel() + att.numel() + proj.numel() + out.numel()
    print(f"total states: {total_states}")
    for name, param in attention.named_parameters():
        print(f"{name}: {param.shape}")
    
    # fakeloss
    dout = torch.randn_like(out) / (B * (H * W)**0.5)
    fakeloss = torch.sum(out * dout)
    fakeloss.backward()
    
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 12345678 # magic number
    header[1] = B
    header[2] = C
    header[3] = H
    header[4] = W
    header[5] = HS
    header[6] = gn_n_groups
    
    with open("attention_block_params.bin", "wb") as f:
        f.write(header.numpy().tobytes())
        for name, param in attention.named_parameters():
            print(f"writing {name}, shape {param.shape}")
            write(param, f)

    with open("attention_block_states.bin", "wb") as f:
        write(x, f)
        write(gn, f)
        write(perm1, f)
        write(qkv, f)
        write(att, f)
        write(proj, f)
        write(out, f)
        write(dout, f)
        write(x.grad, f)
        for _, param in attention.named_parameters():
            write(param.grad, f)
        