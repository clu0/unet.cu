import torch
from utils import write


N = 8 * 64
H = 16
W = 16

#x = torch.arange(N).float().view(N, 1, 1).requires_grad_(True)
x = torch.randn(N, 1, 1, requires_grad=True)
out = x.expand(N, H, W)

#dout = torch.arange(N * H * W) * 0.1
dout = torch.randn(N * H * W) * ((H * W)**-0.5)
fakeloss = (out * dout.view(N, H, W)).sum()
fakeloss.backward()

with open("broadcast.bin", "wb") as file:
    write(x, file)
    write(out, file)
    write(dout, file)
    write(x.grad, file)