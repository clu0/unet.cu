"""
manually implement the forward and backward passes for 2d downsample with stride 2
"""
import torch
import torch.nn as nn
from utils import write


torch.manual_seed(0)
class AvgPool2D:
    """
    expect input x to be of shape (B, C, H, W)

    Note: there is implicit zero padding if H or W are not even
    """
    
    @staticmethod
    def forward(x: torch.Tensor):
        h_sum = x[:, :, ::2, :] + x[:, :, 1::2, :]
        wh_sum = h_sum[:, :, :, ::2] + h_sum[:, :, :, 1::2]
        return wh_sum / 4
    
    @staticmethod
    def backward(dout: torch.Tensor):
        B, C, H, W = dout.size()
        expand_h = torch.zeros(B, C, 2*H, W)
        expand_h[:, :, ::2, :] = dout
        expand_h[:, :, 1::2, :] = dout
        dx = torch.zeros(B, C, 2*H, 2*W)
        dx[:, :, :, ::2] = expand_h
        dx[:, :, :, 1::2] = expand_h
        return dx / 4

# create a small dummy example
B = 1
C = 64
H = 32
W = 32
x = torch.randn(B, C, H, W, requires_grad=True)
out = AvgPool2D.forward(x)

# compare out from the output of nn.AvgPool2d
avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
out_torch = avgpool(x)
print("out error:", (out - out_torch).abs().max().item())

dout = torch.randn(B, C, H//2, W//2)
dx = AvgPool2D.backward(dout)

# compare to PyTorch autograd
fakeloss = (out * dout).sum()
fakeloss.backward()
print("dx error:", (x.grad - dx).abs().max().item())


# Write to file
with open('down.bin', 'wb') as file:
    write(x, file) # (B, C, H, W)
    write(out_torch, file)
    write(dout, file)
    write(x.grad, file)