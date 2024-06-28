#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include "silu.cuh"



// ----------------------------------------------------------------------------
// GPU kernels


__global__ void silu_forward_kernel(
    const float* x,
    float* out,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {
        return;
    }

    float x_val = x[idx];
    out[idx] = x_val / (1.0f + expf(-x_val));
}


__global__ void silu_backward_kernel(
    const float* dout, const float* x,
    float* dx,
    int N
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) {
        return;
    }
    float out_val = dout[idx];
    float x_val = x[idx];
    float expx = expf(-x_val);
    float grad_silu = (1.0f + x_val * expx / (1.0f + expx)) / (1.0f + expx);
    dx[idx] = out_val * grad_silu;
}


// ----------------------------------------------------------------------------
// CUDA kernel launcher

void silu_forward(
    const float* x,
    float* out,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    silu_forward_kernel<<<n_blk, block_size>>>(x, out, N);
}

void silu_backward(
    const float* dout, const float* x,
    float* dx,
    int N, int block_size
) {
    int n_blk = ceil_div(N, block_size);
    silu_backward_kernel<<<n_blk, block_size>>>(dout, x, dx, N);
}

// ----------------------------------------------------------------------------

#ifndef LINKING
int main(int argc, char **argv) {
    int B = 1;
    int C = 32;
    int H = 32;
    int W = 32;
    int N = B * C * H * W;

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // create host memory and load data
    float* x = (float*)malloc(N * sizeof(float));
    float* out = (float*)malloc(N * sizeof(float));
    float* dout = (float*)malloc(N * sizeof(float));
    float* dx = (float*)malloc(N * sizeof(float));

    // read saved output
    FILE *file = fopenCheck("silu.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }

    freadCheck(x, sizeof(float), N, file);
    freadCheck(out, sizeof(float), N, file);
    freadCheck(dout, sizeof(float), N, file);
    freadCheck(dx, sizeof(float), N, file);
    fcloseCheck(file);


    // allocate device memory
    float *d_x, *d_out, *d_dout, *d_dx;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, N * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {128, 256, 512, 1024};
    printf("Checking forward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);
        silu_forward(d_x, d_out, N, block_size);
        validate_result(d_out, out, "out", N);
    }
    printf("Forward pass: all results match\n\n");

    printf("Checking backward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);
        silu_backward(d_dout, d_x, d_dx, N, block_size);
        validate_result(d_dx, dx, "dx", N);
    }
    printf("Backward pass: all results match\n\n");
    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, silu_forward,
                                              d_x, d_out, N, block_size);

        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    printf("\nBackward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, silu_backward,
                                              d_dout, d_x, d_dx,
                                              N, block_size);

        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }

    // free memory
    free(x);
    free(out);
    free(dout);
    free(dx);

    cudaCheck(cudaFree(d_x));
    cudaCheck(cudaFree(d_out));
    cudaCheck(cudaFree(d_dout));
    cudaCheck(cudaFree(d_dx));
}
#endif