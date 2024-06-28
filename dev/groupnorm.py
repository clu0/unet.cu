import torch
import torch.nn as nn
from utils import write


torch.manual_seed(0)


# create and save a dummy example
#B = 32
#C = 64
#H = 64
#W = 64
B = 16
C = 128
H = 4
W = 8
img_size = H * W

# use a handcrafted example for debug
# x = (torch.arange(B * C * H * W) * 0.0001).reshape(B, C, H, W).clone().detach().requires_grad_(True)
# weight = nn.Parameter(torch.arange(C).float().clone().detach().requires_grad_(True))
# bias = nn.Parameter(torch.arange(C).float().clone().detach().requires_grad_(True))
x = (torch.randn(B, C, H, W) * (img_size **-0.5)).requires_grad_(True)
weight = nn.Parameter((torch.randn(C) * C**-0.5).requires_grad_(True))
bias = nn.Parameter((torch.randn(C) * C**-0.5).requires_grad_(True))
norm = nn.GroupNorm(32, C)
norm.weight = weight
norm.bias = bias
out = norm(x)

# dout = torch.arange(B * C * H * W).reshape(B, C, H, W) * 0.0001
dout = (torch.randn(B, C, H, W) * (img_size **-0.5)).requires_grad_(True)
fakeloss = (out * dout).sum()
fakeloss.backward()

#print(f"x: {x},\nout: {out},\ndout: {dout},\nx.grad: {x.grad}")
#print(f"weight.grad: {weight.grad}, bias.grad: {bias.grad}")

with open('groupnorm.bin', 'wb') as file:
    write(x, file)
    write(weight, file)
    write(bias, file)
    write(out, file)
    write(dout, file)
    write(x.grad, file)
    write(weight.grad, file)
    write(bias.grad, file)