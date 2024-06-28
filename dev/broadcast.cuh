#pragma once


void broadcast_last_dims_forward(
    const float* x,
    float* out,
    int N, int H, int W, int block_size
);

void broadcast_last_dims_backward(
    const float* dout,
    float* dx,
    int N, int H, int W, int block_size
);