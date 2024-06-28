import torch
import torch.nn as nn
from utils import write

torch.manual_seed(0)

# create and save a dummy example
B = 1
C = 32
H = 32
W = 32
x = torch.randn(B, C, H, W, requires_grad=True)

# use a handcrafted example for debug
# shape (1, 2, 3, 4)
#x = (torch.arange(24).reshape(1, 2, 3, 4) * 0.1).clone().detach().requires_grad_(True)
silu = nn.SiLU()
out = silu(x)

# dout = torch.arange(24).reshape(1, 2, 3, 4) * 0.1
dout = torch.randn(B, C, H, W)
fakeloss = (out * dout).sum()
fakeloss.backward()

# print(f"x: {x},\nout: {out},\ndout: {dout},\nx.grad: {x.grad}")

with open('silu.bin', 'wb') as file:
    write(x, file)
    write(out, file)
    write(dout, file)
    write(x.grad, file)