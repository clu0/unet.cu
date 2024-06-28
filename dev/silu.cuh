#pragma once

void silu_forward(
    const float* x,
    float* out,
    int N, int block_size
);

void silu_backward(
    const float* dout, const float* x,
    float* dx,
    int N, int block_size
);

typedef struct {
    float* x;
    float* out;
} SiluActs;
