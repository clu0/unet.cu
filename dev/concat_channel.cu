#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include "concat_channel.cuh"


__global__ void concat_channel_forward_kernel(
    const float* x1, const float* x2,
    float* out,
    int B, int C1, int C2, int H, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C1 * H * W) {
        // copy input from x1
        int b = idx / (C1 * H * W);
        int c = (idx / H / W) % C1;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + c * H * W + h * W + w;
        out[out_idx] = x1[idx];
    }
    if (idx < B * C2 * H * W) {
        // copy input from x2
        // move over from x1
        int b = idx / (C2 * H * W);
        int c = (idx / H / W) % C2;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + (C1 + c) * H * W + h * W + w;
        
        out[out_idx] = x2[idx];
    }
}

__global__ void concat_channel_backward_kernel(
    const float* dout,
    float* dx1, float* dx2,
    int B, int C1, int C2, int H, int W
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < B * C1 * H * W) {
        int b = idx / (C1 * H * W);
        int c = (idx / H / W) % C1;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + c * H * W + h * W + w;

        dx1[idx] = dout[out_idx];
    }
    if (idx < B * C2 * H * W) {
        int b = idx / (C2 * H * W);
        int c = (idx / H / W) % C2;
        int h = (idx / W) % H;
        int w = idx % W;
        int out_idx = b * (C1 + C2) * H * W + (C1 + c) * H * W + h * W + w;
        
        dx2[idx] = dout[out_idx];
    }
}

void concat_channel_forward(
    const float* x1, const float* x2,
    float* out,
    int B, int C1, int C2, int H, int W, int block_size
) {
    int N = B * max_int(C1, C2) * H * W;
    int n_blk = ceil_div(N, block_size);
    concat_channel_forward_kernel<<<n_blk, block_size>>>(x1, x2, out, B, C1, C2, H, W);
}

void concat_channel_backward(
    const float* dout,
    float* dx1, float* dx2,
    int B, int C1, int C2, int H, int W, int block_size
) {
    int N = B * max_int(C1, C2) * H * W;
    int n_blk = ceil_div(N, block_size);
    concat_channel_backward_kernel<<<n_blk, block_size>>>(dout, dx1, dx2, B, C1, C2, H, W);
}