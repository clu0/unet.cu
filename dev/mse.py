import torch
import torch.nn as nn
from utils import write

torch.manual_seed(0)

N = 32 * 3 * 64 * 64

x = torch.randn(N).requires_grad_(True)
y = torch.randn(N)

mse = (x - y).pow(2).mean()
mse.backward()

with open("mse.bin", "wb") as file:
    write(x, file)
    write(y, file)
    write(x.grad, file)
    write(mse, file)