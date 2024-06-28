#pragma once


void concat_channel_forward(
    const float* x1, const float* x2,
    float* out,
    int B, int C1, int C2, int H, int W, int block_size
);


void concat_channel_backward(
    const float* dout,
    float* dx1, float* dx2,
    int B, int C1, int C2, int H, int W, int block_size
);

typedef struct {
    float* x1;
    float* x2;
    float* out;
} ConcatChannelActs;