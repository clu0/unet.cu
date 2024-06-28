import torch
import torch.nn as nn
from utils import write

torch.manual_seed(0)

device = 'cuda:1'
# create a small dummy example
B = 32
C_in = 64
C_out = 128

x = torch.randn(B, C_in, requires_grad=True, device=device)
w = nn.Parameter(torch.randn(C_out, C_in, requires_grad=True, device=device))
bias = nn.Parameter(torch.randn(C_out, requires_grad=True, device=device))

linear = nn.Linear(C_in, C_out)
linear.weight = w
linear.bias = bias

out = linear(x)

dout = torch.randn(B, C_out, device=device)
fakeloss = (out * dout).sum()
fakeloss.backward()
print("x.grad sum :", x.grad.sum().item())
print("w.grad sum :", w.grad.sum().item())
print("bias.grad sum :", bias.grad.sum().item())

with open('linear.bin', 'wb') as file:
    write(x, file)
    write(w, file)
    write(bias, file)
    write(out, file)
    write(dout, file)
    write(x.grad, file)
    write(w.grad, file)
    write(bias.grad, file)