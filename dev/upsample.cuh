#pragma once

void upsample_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size
);

void upsample_backward1(
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size
);


typedef struct {
    float* x;
    float* out;
} UpsampleActs;