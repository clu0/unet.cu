#pragma once

void add_forward(
    const float* a, const float* b,
    float* out,
    int N, int block_size
);


void add_inplace_forward(
    const float* a, float* b,
    int N, int block_size
);