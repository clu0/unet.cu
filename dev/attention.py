import torch
import torch.nn as nn
from utils import write


def attention_forward(qkv: torch.Tensor, HS: int):
    """
    input tensor shape (B, T, 3*C)
    """
    B, T, C3 = qkv.size()
    C = C3 // 3
    assert C % HS == 0
    NH = C // HS
    q, k, v = qkv.chunk(3, dim=-1)
    q = q.view(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()
    k = k.view(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()
    v = v.view(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()
    scale = HS ** -0.5

    preattn = torch.einsum("bhnd,bhmd->bhnm", q, k) * scale
    attn = torch.softmax(preattn, dim=-1)
    out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
    out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
    return out


device = "cuda:1"
B = 4
T = 1024
C = 256
HS = 32
qkv = torch.randn(B, T, 3*C, requires_grad=True, device=device)
out = attention_forward(qkv, HS)

# fakeloss
dout = torch.randn(B, T, C, device=device)
fakeloss = (out * dout).sum()
fakeloss.backward()

with open('attention.bin', 'wb') as file:
    write(qkv, file)
    write(out, file)
    write(dout, file)
    write(qkv.grad, file)