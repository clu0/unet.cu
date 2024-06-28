#pragma once


void avgpool_2d_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    const int block_size
);

void avgpool_2d_backward1(
    const float* dout,
    float* dx,
    int B, int C, int H, int W,
    const int block_size
);

typedef struct {
    float* x;
    float* out;
} AvgpoolActs;