import torch
import torch.nn as nn
from utils import write


torch.manual_seed(0)


class Conv2d_k1:
    """
    Conv2d with kernel size 1
    """

    @staticmethod
    def forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        """
        x shape  (B, C_in, H, W)
        w shape (C_out, C_in, 1, 1)
        bias shape (C_out,)
        """
        B, C_in, H, W = x.size()
        C_out = w.size(0)
        # transpose x from (B, C_in, H, W) into (B, W, H, C_in)
        x_t = x.transpose(1, 3).reshape(-1, C_in).contiguous()
        # transpose W from (C_out, C_in, 1, 1) into (C_in, C_out)
        w_t = w.transpose(0, 1).view(C_in, C_out).contiguous()
        out_t = x_t @ w_t + b

        # out_t shape (B, W, H, C_out), transpose into (B, C_out, H, W)
        out = out_t.reshape(B, W, H, C_out).transpose(1, 3).contiguous()
        return out
    
    @staticmethod
    def backward(dout: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
        """
        shapes:
        dout: (B, C_out, H, W)
        x: (B, C_in, H, W)
        w: (C_out, C_in, 1, 1)
        """
        C_in = x.size(1)
        C_out = w.size(0)
        db = dout.sum((0, 2, 3))
        
        # transpose dout to shape (B, H, W, C_out)
        dout_t = dout.permute(0, 2, 3, 1).contiguous()
        # transpose x to shape (B, H, W, C_in)
        x_t = x.permute(0, 2, 3, 1).contiguous()
        
        # compute dw: reshape dout_t to (B*H*W, C_out), and x_t to (B*H*W, C_in)
        dout_flat = dout_t.reshape(-1, C_out)
        x_flat = x_t.reshape(-1, C_in)
        dw = dout_flat.T @ x_flat
        dw = dw.reshape(C_out, C_in, 1, 1)
        
        # compute dx: reshape w to (1, 1, C_out, C_in)
        w = w.view(1, 1, C_out, C_in)
        dx_t = dout_t @ w
        # permute dx from (B, H, W, C_in) to (B, C_in, H, W)
        dx = dx_t.permute(0, 3, 1, 2).contiguous()

        return dx, dw, db


# create a small dummy example
B = 32
C_in = 64
C_out = 128
H = 64
W = 64
x = torch.randn(B, C_in, H, W, requires_grad=True)
w = nn.Parameter(torch.randn(C_out, C_in, 1, 1, requires_grad=True))
b = nn.Parameter(torch.randn(C_out, requires_grad=True))

# compare to output from nn.Conv2d
conv = nn.Conv2d(C_in, C_out, kernel_size=1)
conv.weight = w
conv.bias = b
out_torch = conv(x)

out = Conv2d_k1.forward(x, w, b)

print("out error:", (out - out_torch).abs().max().item())

dout = torch.randn(B, C_out, H, W)
dx, dw, db = Conv2d_k1.backward(dout, x, w)

# compare to PyTorch autograd
fakeloss = (out * dout).sum()
fakeloss.backward()
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())


with open('conv2d_k1.bin', 'wb') as file:
    write(x, file)
    write(w, file)
    write(b, file)
    write(out_torch, file)
    write(dout, file)
    write(x.grad, file)
    write(w.grad, file)
    write(b.grad, file)