import torch
import torch.nn as nn
from utils import write
from time import time
from tqdm import tqdm


torch.manual_seed(0)

device = 'cuda:1'
class Conv2d_k3:
    """
    Conv2d with kernel size 3
    """
    @staticmethod
    def forward(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor):
        """
        x shape  (B, C_in, H, W)
        w shape (C_out, C_in, 3, 3)
        bias shape (C_out,)
        """
        B, C_in, H, W = x.size()
        C_out = w.size(0)
        # create a (B, C_in, H, W, 9) tensor
        # with padded views of x
        x_copies = torch.zeros(B, C_in, H, W, 9, device=device)
        for ind in range(9):
            i = ind // 3
            j = ind % 3
            i_start_c = max(0, 1 - i)
            i_end_c = min(H, H + 1 - i)
            i_start_o = max(0, i - 1)
            i_end_o = min(H, H + i - 1)
            j_start_c = max(0, 1 - j)
            j_end_c = min(W, W + 1 - j)
            j_start_o = max(0, j - 1)
            j_end_o = min(W, W + j - 1)
            x_copies[:, :, i_start_c:i_end_c, j_start_c:j_end_c, ind] = x[:, :, i_start_o:i_end_o, j_start_o:j_end_o]
        
        # permute x_copies to (B, H, W, C_in*9)
        x_copies = x_copies.permute(0, 2, 3, 1, 4).contiguous().view(B, H, W, C_in*9)
        # permute w to (1, 1, C_out, C_in*9)
        w_flat = w.view(1, 1, C_out, C_in*9)

        out = x_copies @ w_flat.transpose(2, 3) + b.view(1, 1, 1, C_out)
        # permute out from (B, H, W, C_out) to (B, C_out, H, W)
        out = out.permute(0, 3, 1, 2).contiguous()
        return out
    
    @staticmethod
    def backward(dout: torch.Tensor, x: torch.Tensor, w: torch.Tensor):
        """
        shapes:
        dout: (B, C_out, H, W)
        x: (B, C_in, H, W)
        w: (C_out, C_in, 3, 3)
        """
        db = dout.sum((0, 2, 3))
        B, C_in, H, W = x.size()
        C_out = w.size(0)

        dx = torch.zeros_like(x, device=device)
        dw = torch.zeros_like(w, device=device)

        # permute dout from (B, C_out, H, W) to (B, H, W, C_out)
        dout_perm = dout.permute(0, 2, 3, 1).contiguous()
        
        x_tiled = torch.zeros(B*H*W, C_in, 9, device=device)

        # iteratively compute dx and dw corresponding to each element of the 3x3 kernel
        for ind in range(9):
            i = ind // 3
            j = ind % 3
            i_start_c = max(0, 1 - i)
            i_end_c = min(H, H + 1 - i)
            i_start_o = max(0, i - 1)
            i_end_o = min(H, H + i - 1)
            j_start_c = max(0, 1 - j)
            j_end_c = min(W, W + 1 - j)
            j_start_o = max(0, j - 1)
            j_end_o = min(W, W + j - 1)

            x_slice = torch.zeros_like(x)
            x_slice[:, :, i_start_c:i_end_c, j_start_c:j_end_c] = x[:, :, i_start_o:i_end_o, j_start_o:j_end_o]
            w_slice = w[:, :, i, j]

            # dw
            # permute x_slice from (B, C_in, H, W) to (B*H*W, C_in)
            x_slice = x_slice.permute(0, 2, 3, 1).contiguous().view(B*H*W, C_in)
            x_tiled[:, :, ind] = x_slice
            #dw[:, :, i, j] = dout_perm.view(B*H*W, C_out).T @ x_slice


            # permute w_slice from (C_out, C_in) to (1, 1, C_out, C_in)
            w_slice = w_slice.view(1, 1, C_out, C_in)
            dx_tmp = dout_perm @ w_slice
            # this part is tricky: dx_tmp is not the contribution to dx from this element of the kernel
            # because not all elements of x would have been multiplied with the (i, j) element of the kernel
            # so we create dx_slice, which only has non-zero values for the indices of x that were multiplied
            # with the (i, j) element of the kernel.
            # e.g. for (i, j) = (0, 0), then the coordinate (h, w) = (0, 0) would not have been multiplied with the kerel element
            # so dx_slice[:, 0, 0, :] = 0
            dx_slice = torch.zeros_like(dx_tmp, device=device)
            dx_slice[:, i_start_o:i_end_o, j_start_o:j_end_o, :] = dx_tmp[:, i_start_c:i_end_c, j_start_c:j_end_c, :]
            # permute from (B, H, W, C_in) to (B, C_in, H, W)
            dx_slice = dx_slice.permute(0, 3, 1, 2).contiguous()
            dx += dx_slice
        dout_perm = dout_perm.permute(3, 0, 1, 2).contiguous()
        dout_perm = dout_perm.view(C_out, B*H*W)
        x_tiled = x_tiled.view(B*H*W, C_in*9)
        dw = dout_perm @ x_tiled
        dw = dw.view(C_out, C_in, 3, 3)
        
        return dx, dw, db, dout_perm, x_tiled
            



# create a small dummy example
B = 32
C_in = 192
C_out = 64
H = 64
W = 64
x = torch.randn(B, C_in, H, W, requires_grad=True, device='cuda:1')
w = nn.Parameter(torch.randn(C_out, C_in, 3, 3, requires_grad=True, device='cuda:1'))
b = nn.Parameter(torch.randn(C_out, requires_grad=True, device='cuda:1'))

# compare to output from nn.Conv2d
conv = nn.Conv2d(C_in, C_out, kernel_size=3, padding=1)
conv.weight = w
conv.bias = b
out_torch = conv(x)

out = Conv2d_k3.forward(x, w, b)
print("out error:", (out - out_torch).abs().max().item())

dout = torch.randn(B, C_out, H, W, device='cuda:1')
dx, dw, db, dout_perm, x_tiled = Conv2d_k3.backward(dout, x, w)
print("x_tiled sum:", x_tiled.sum())

# compare to PyTorch autograd
fakeloss = (out * dout).sum()
fakeloss.backward()
print("dx error:", (x.grad - dx).abs().max().item())
print("dw error:", (w.grad - dw).abs().max().item())
print("db error:", (b.grad - db).abs().max().item())

with open('conv2d_k3.bin', 'wb') as file:
    write(x, file)
    write(w, file)
    write(b, file)
    write(out_torch, file)
    write(dout, file)
    write(x.grad, file)
    write(w.grad, file)
    write(b.grad, file)
    write(dout_perm, file)
    write(x_tiled, file)

# some quick benchmarks
forward_start = torch.cuda.Event(enable_timing=True)
forward_end = torch.cuda.Event(enable_timing=True)
full_start = torch.cuda.Event(enable_timing=True)
full_end = torch.cuda.Event(enable_timing=True)
n_reps = 5000
forward_time = 0.0
backward_time = 0.0
torch.cuda.synchronize()
forward_start.record()
for i in tqdm(range(n_reps)):
    out_torch = conv(x)
torch.cuda.synchronize()
forward_end.record()
torch.cuda.synchronize()

full_start.record()
for i in tqdm(range(n_reps)):
    out_torch = conv(x)
    loss_torch = (out_torch * dout).sum()
    loss_torch.backward()
    conv.zero_grad()
torch.cuda.synchronize()
full_end.record()
torch.cuda.synchronize()

forward_time = forward_start.elapsed_time(forward_end)
full_pass_time = full_start.elapsed_time(full_end)
backward_time = full_pass_time - forward_time

print(f"forward avg time in ms: {forward_time/n_reps}")
print(f"backward avg time in ms: {backward_time/n_reps}")
