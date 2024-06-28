#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include "avgpool.cuh"

// ----------------------------------------------------------------------------
// GPU kernels

__global__ void avgpool_2d_forward_kernel(
    const float* x,  // (B, C, H, W)
    float* out, // (B, C, H/2, W/2)
    int B, int C, int H, int W
) {
    //printf("started kernel\n");
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int H_2 = H / 2;
    int W_2 = W / 2;
    int img_out_size = H_2 * W_2;
    if (flat_idx >= B * C * img_out_size) {
        return;
    }
    int img_size = H * W;
    int b = flat_idx / (C * img_out_size);
    int c = (flat_idx / img_out_size) % C;
    int i = (flat_idx / W_2) % H_2;
    int j = flat_idx % W_2;

    // move pointers
    x += b * C * img_size + c * img_size + 2 * i * W + 2 * j;
    out += b * C * img_out_size + c * img_out_size + i * W_2 + j;

    float sum = 0.0f;
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            sum += x[ii * W + jj];
        }
    }
    out[0] = sum / 4.0f;
}


__global__ void avgpool_2d_backward_kernel(
    const float* dout,
    float* dx,
    int B, int C, int H_out, int W_out // H_out and W_out are the shape of the output from forward, i.e. half the original sizes
) {
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int img_out_size = H_out * W_out;
    if (flat_idx >= B * C * img_out_size) {
        return;
    }
    int H_in = H_out * 2;
    int W_in = W_out * 2;
    int img_in_size = H_in * W_in;
    int b = flat_idx / (C * img_out_size);
    int c = (flat_idx / img_out_size) % C;
    int i = (flat_idx / W_out) % H_out;
    int j = flat_idx % W_out;

    // move pointers
    dout += b * C * img_out_size + c * img_out_size + i * W_out + j;
    dx += b * C * img_in_size + c * img_in_size + 2 * i * W_in + 2 * j;

    float d = dout[0] / 4.0f;

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            dx[ii * W_in + jj] = d;
        }
    }
}

// ----------------------------------------------------------------------------
// kernel launcher

void avgpool_2d_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    const int block_size
) {
    int gridDim = ceil_div(B * C * (H / 2) * (W / 2), block_size);
    avgpool_2d_forward_kernel<<<gridDim, block_size>>>(x, out, B, C, H, W);
    cudaCheck(cudaGetLastError());
}

void avgpool_2d_backward1(
    const float* dout,
    float* dx,
    int B, int C, int H, int W,
    const int block_size
) {
    int H_out = H / 2;
    int W_out = W / 2;
    int gridDim = ceil_div(B * C * H_out * W_out, block_size);
    avgpool_2d_backward_kernel<<<gridDim, block_size>>>(dout, dx, B, C, H_out, W_out);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------

#ifndef LINKING
int main(int argc, char **argv) {
    srand(0);
    int B = 1;
    int C = 64;
    int H = 32;
    int W = 32;

    // set up the device
    int deviceIdx = 0;
    cudaCheck(cudaSetDevice(deviceIdx));
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, deviceIdx);
    printf("Device %d: %s\n", deviceIdx, deviceProp.name);

    // create host memory to load data
    float* x = (float*)malloc(B * C * H * W * sizeof(float));
    float* out = (float*)malloc(B * C * H/2 * W/2 * sizeof(float));
    float* dx = (float*)malloc(B * C * H * W * sizeof(float));
    float* dout = (float*)malloc(B * C * H/2 * W/2 * sizeof(float));

    // read saved output
    FILE *file = fopen("down.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(x, sizeof(float), B * C * H * W, file);
    freadCheck(out, sizeof(float), B * C * H/2 * W/2, file);
    freadCheck(dout, sizeof(float), B * C * H/2 * W/2, file);
    freadCheck(dx, sizeof(float), B * C * H * W, file);
    fclose(file);


    // allocate device memory
    float *d_x, *d_out, *d_dout, *d_dx;
    cudaCheck(cudaMalloc(&d_x, B * C * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * C * H/2 * W/2 * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * C * H/2 * W/2 * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, B * C * H * W * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * C * H/2 * W/2 * sizeof(float), cudaMemcpyHostToDevice));

    printf("Checking forward pass\n");
    int block_sizes[] = {32, 64, 128};
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);
        avgpool_2d_forward1(d_out, d_x, B, C, H, W, block_size);
        validate_result(d_out, out, "out", B * C * (H/2) * (W/2));
    }
    printf("Forward pass: all results match\n\n");

    printf("Checking backward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);
        avgpool_2d_backward1(d_dout, d_dx, B, C, H, W, block_size);
        validate_result(d_dx, dx, "dx", B * C * H * W);
    }
    printf("Backward pass: all results match\n\n");
    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, avgpool_2d_forward1,
                                              d_out, d_x,
                                              B, C, H, W, block_size);

        float tflops = (float)B * H * W * C / elapsed_time * 1e3f / 1e12f;
        printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
    }

    printf("\nBackward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, avgpool_2d_backward1,
                                              d_dout, d_dx,
                                              B, C, H, W, block_size);

        float tflops = (float)B * H * W * C / elapsed_time * 1e3f / 1e12f;
        printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
    }

    // free memory
    free(x);
    free(out);
    free(dout);
    free(dx);
    cudaFree(d_x);
    cudaFree(d_out);
    cudaFree(d_dout);
    cudaFree(d_dx);

    return 0;
}
#endif