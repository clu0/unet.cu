#pragma once


#include <cublas_v2.h>
#include <cuda_runtime.h>


void conv2d_k3_forward2(
    cublasHandle_t cublas_handle,
    const float* x, const float* weight, const float* bias,
    float* out,
    int B, int C_in, int C_out, int H, int W,
    int block_size,
    float* t1 = nullptr, float* t2 = nullptr, float* t3 = nullptr
);

void conv2d_k3_forward3(
    float* x, float* weight, float* bias,
    float* out,
    const int B, const int C_in, const int C_out, const int H, const int W
);

void conv2d_k3_backward1(
    cublasHandle_t cublas_handle,
    const float* dout, const float* x, const float* weight,
    float* dx, float* dweight, float* dbias,
    int B, int C_in, int C_out, int H, int W,
    int block_size,
    float* t1 = nullptr, float* t2 = nullptr, float* t3 = nullptr, float* t4 = nullptr, float* t5 = nullptr,
    float* t6 = nullptr
);

void conv2d_k3_backward2(
    float* dout, float* x, float* weight,
    float* dweight_buf, float* dbias_buf,
    float* dx, float* dweight, float* dbias,
    const int B, const int C_in, const int C_out, const int H, const int W
);

typedef struct {
    float* w;
    float* b;
} ConvK3Params;

inline void convk3_set_param_ptrs(
    ConvK3Params* params,
    float* params_memory,
    int C_in, int C_out
) {
    params->w = params_memory;
    params->b = params->w + C_in * C_out * 9;
}

inline size_t convk3_count_params(int C_in, int C_out) {
    return C_in * C_out * 9 + C_out;
}

typedef struct {
    float* inp;
    float* out;
} ConvK3Acts;