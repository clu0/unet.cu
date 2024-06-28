#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include "common.h"
#include "broadcast.cuh"

// kernel to broadcast a tensor from shape (B, C) to (B, C, H, W)
// Assuming one thread per pixel
__global__ void broadcast_last_dims_forward_kernel(
    const float* x,
    float* out,
    int N, int H, int W, int n_bk_per_img
) {
    int n = blockIdx.x / n_bk_per_img;
    __shared__ float x_sh[1];
    if (threadIdx.x == 0) {
        x_sh[0] = x[n];
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N * H * W) {
        out[idx] = x_sh[0];
    }
}

void broadcast_last_dims_forward(
    const float* x,
    float* out,
    int N, int H, int W, int block_size
) {
    block_size = min_int(block_size, H * W);
    int n_bk_per_img = ceil_div(H * W, block_size);
    broadcast_last_dims_forward_kernel<<<N * n_bk_per_img, block_size>>>(x, out, N, H, W, n_bk_per_img);
};

// backwards for broadcast
// use one warp per image
__global__ void broadcast_last_dims_backward_kernel(
    const float* dout,
    float* dx,
    int N, int H, int W
) {
    namespace cg = cooperative_groups;
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    int idx = blockIdx.x * warp.meta_group_size() + warp.meta_group_rank();
    if (idx >= N) { return; }

    int img_size = H * W;
    dout += idx * img_size;
    dx += idx;

    float sum = 0.0f;
    for (int i = warp.thread_rank(); i < img_size; i += warp.size()) {
        sum += dout[i];
    }
    sum = cg::reduce(warp, sum, cg::plus<float>());
    if (warp.thread_rank() == 0) {
        dx[0] = sum;
    }
}

void broadcast_last_dims_backward(
    const float* dout,
    float* dx,
    int N, int H, int W, int block_size
) {
    block_size = min_int(block_size, H * W);
    assert(block_size % 32 == 0);
    int warps_per_block = block_size / 32;
    int gridDim = ceil_div(N, warps_per_block);
    broadcast_last_dims_backward_kernel<<<gridDim, block_size>>>(dout, dx, N, H, W);
};


#ifndef LINKING
int main(int argc, char **argv) {
    int N = 8 * 64;
    int H = 16;
    int W = 16;

    // create host memory
    float *x = (float*)malloc(N * sizeof(float));
    float *out = (float*)malloc(N * H * W * sizeof(float));
    float *dout = (float*)malloc(N * H * W * sizeof(float));
    float *dx = (float*)malloc(N * sizeof(float));

    // read saved output
    FILE *file = fopen("broadcast.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(x, sizeof(float), N, file);
    freadCheck(out, sizeof(float), N * H * W, file);
    freadCheck(dout, sizeof(float), N * H * W, file);
    freadCheck(dx, sizeof(float), N, file);
    fclose(file);

    // allocate device memory
    float *d_x, *d_out, *d_dout, *d_dx;
    cudaCheck(cudaMalloc(&d_x, N * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, N * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, N * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, N * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, N * H * W * sizeof(float), cudaMemcpyHostToDevice));

    int block_sizes[] = {512};
    printf("Checking forward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("\nChecking block size %d\n", block_size);
        broadcast_last_dims_forward(d_x, d_out, N, H, W, block_size);
        validate_result(d_out, out, "out", N * H * W);
    }
    printf("Checking backward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("\nChecking block size %d\n", block_size);
        broadcast_last_dims_backward(d_dout, d_dx, N, H, W, block_size);
        validate_result(d_dx, dx, "dx", N);
    }


    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, broadcast_last_dims_forward,
            d_out, d_x, N, H, W, block_size);
        
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }
    printf("Backward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, broadcast_last_dims_backward,
            d_dout, d_dx, N, H, W, block_size);
        
        printf("block_size %4d | time %.4f ms\n", block_size, elapsed_time);
    }
}
#endif