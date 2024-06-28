#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

void conv2d_k1_forward1(
    cublasHandle_t cublas_handle,
    float* out,
    const float* x, const float* weight, const float* bias,
    int B, int C_in, int H, int W, int C_out,
    const int block_size, 
    float* t1 = nullptr, float* t2 = nullptr, float* t3 = nullptr
);

void conv2d_k1_forward2(
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C_in, int H, int W, int C_out
);

void conv2d_k1_backward1(
    cublasHandle_t cublas_handle,
    const float* dout, const float* x, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C_in, int C_out, int H, int W,
    int block_size
);