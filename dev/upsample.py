import torch
import torch.nn.functional as F
from utils import write

torch.manual_seed(0)


class Upsample:
    """
    nearst neighbor scale up by factor of 2
    """
    @staticmethod
    def forward(x: torch.Tensor):
        B, C, H, W = x.size()
        out = torch.zeros(B, C, H*2, W*2)
        out[:, :, ::2, ::2] = x
        out[:, :, 1::2, ::2] = x
        out[:, :, ::2, 1::2] = x
        out[:, :, 1::2, 1::2] = x

        return out
    
    @staticmethod
    def backward(dout: torch.Tensor):
        h_sum = dout[:, :, ::2, :] + dout[:, :, 1::2, :]
        wh_sum = h_sum[:, :, :, ::2] + h_sum[:, :, :, 1::2]
        return wh_sum



# create a small dummy example
B = 1
C = 64
H = 32
W = 32
x = torch.randn(B, C, H, W, requires_grad=True)
out = Upsample.forward(x)

# compare to output from F.interpolate
out_torch = F.interpolate(x, scale_factor=2, mode="nearest")

print("out error:", (out - out_torch).abs().max().item())

dout = torch.randn(B, C, 2*H, 2*W)
dx = Upsample.backward(dout)

# compare to PyTorch autograd
fakeloss = (out * dout).sum()
fakeloss.backward()
print("dx error:", (x.grad - dx).abs().max().item())


with open('upsample.bin', 'wb') as file:
    write(x, file)
    write(out_torch, file)
    write(dout, file)
    write(x.grad, file)