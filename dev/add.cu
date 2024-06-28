#include <cuda_runtime.h>
#include "add.cuh"
#include "common.h"

// simple kernel to add two tensors
__global__ void add_forward_kernel(
    const float* a, const float* b,
    float* out,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        out[idx] = a[idx] + b[idx];
    }
}

void add_forward(
    const float* a, const float* b,
    float* out,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    add_forward_kernel<<<n_blk, block_size>>>(a, b, out, N);
}


// add A and B in place, and store result in B
__global__ void add_inplace_forward_kernel(
    const float* a, float* b,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N) {
        b[idx] += a[idx];
    }
}

void add_inplace_forward(
    const float* a, float* b,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    add_inplace_forward_kernel<<<n_blk, block_size>>>(a, b, N);
}