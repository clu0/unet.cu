#pragma once

#include <cublas_v2.h>

void matmul_forward2(
    cublasHandle_t cublas_handle,
    float* out,
    const float* inp, const float* weight, const float* bias,
    int N, int C, int OC,
    const int block_size
);

void matmul_backward1(
    cublasHandle_t cublas_handle,
    float* dinp, float* dweight, float* dbias,
    float* dout, float* inp, float* weight,
    int N, int C, int OC
);

typedef struct {
    float* w; // OC, C
    float* b; // OC
} LinearParams;

typedef struct {
    float* inp; // N, C
    float* out; // N, OC
} LinearActs;


void linear_set_param_ptrs(
    LinearParams* params,
    float* params_memory,
    int C, int OC
);

inline size_t linear_count_params(int C, int OC) {
    return OC * C + OC;
}