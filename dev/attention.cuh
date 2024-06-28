#pragma once

#include <cublas_v2.h>


void attention_forward1
(
    cublasHandle_t cublas_handle,
    float* out, float* qkvr, float* preatt, float* att,
    float* inp,
    int B, int T, int C, int NH,
    const int block_size
);

void attention_backward
(
    cublasHandle_t cublas_handle,
    float* dinp, float* dqkvr, float* dpreatt, float* datt, float* scratch,
    const float* dout,
    const float* qkvr, const float* att,
    int B, int T, int C, int NH
);