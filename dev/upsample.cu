#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "common.h"
#include "upsample.cuh"

// ----------------------------------------------------------------------------
// GPU kernels

// one thread per pixel
__global__ void upsample_forward_kernel(
    const float* x,
    float* out,
    int B, int C, int H, int W
) {
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int img_size = H * W;
    if (flat_idx >= B * C * img_size) {
        return;
    }

    int H_out = H*2;
    int W_out = W*2;
    int img_out_size = H_out * W_out;

    int b = flat_idx / (C * img_size);
    int c = (flat_idx / img_size) % C;
    int i = (flat_idx / W) % H;
    int j = flat_idx % W;

    // move pointers
    x += b * C * img_size + c * img_size + i * W + j;
    out += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

    float x_val = x[0];

    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            out[ii * W_out + jj] = x_val;
        }
    }
}


// number of threads equals the number of pixels
// in downsampled image
__global__ void upsample_backward_kernel(
    const float* dout,
    float* dx,
    int B, int C, int H, int W
) {
    int flat_idx = blockDim.x * blockIdx.x + threadIdx.x;
    int img_in_size = H * W;
    if (flat_idx >= B * C * img_in_size) {
        return;
    }

    int H_out = H*2;
    int W_out = W*2;
    int img_out_size = H_out * W_out;

    int b = flat_idx / (C * img_in_size);
    int c = (flat_idx / img_in_size) % C;
    int i = (flat_idx / W) % H;
    int j = flat_idx % W;

    // move pointers
    dx += b * C * img_in_size + c * img_in_size + i * W + j;
    dout += b * C * img_out_size + c * img_out_size + 2 * i * W_out + 2 * j;

    float dout_sum = 0.0f;
    for (int ii = 0; ii < 2; ii++) {
        for (int jj = 0; jj < 2; jj++) {
            dout_sum += dout[ii * W_out + jj];
        }
    }
    dx[0] = dout_sum;
}

// ----------------------------------------------------------------------------
// kernel launcher
void upsample_forward1(
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size
) {
    int gridDim = ceil_div(B * C * H * W, block_size);
    upsample_forward_kernel<<<gridDim, block_size>>>(x, out, B, C, H, W);
    cudaCheck(cudaGetLastError());
}

void upsample_backward1(
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size
) {
    int gridDim = ceil_div(B * C * H * W, block_size);
    upsample_backward_kernel<<<gridDim, block_size>>>(dout, dx, B, C, H, W);
    cudaCheck(cudaGetLastError());
}

void upsample_forward(
    int kernel_num,
    float* out,
    const float* x,
    int B, int C, int H, int W,
    int block_size
) {
    switch (kernel_num) {
        case 1:
            upsample_forward1(out, x, B, C, H, W, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
}

void upsample_backward(
    int kernel_num,
    float* dx,
    const float* dout,
    int B, int C, int H, int W,
    int block_size
) {
    switch (kernel_num) {
        case 1:
            upsample_backward1(dx, dout, B, C, H, W, block_size);
            break;
        default:
            printf("Invalid kernel number\n");
            exit(1);
    }
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
    float* out = (float*)malloc(B * C * H*2 * W*2 * sizeof(float));
    float* dx = (float*)malloc(B * C * H * W * sizeof(float));
    float* dout = (float*)malloc(B * C * H*2 * W*2 * sizeof(float));

    // read saved output
    FILE *file = fopen("upsample.bin", "rb");
    if (!file) {
        perror("Failed to load data");
        return -1;
    }
    freadCheck(x, sizeof(float), B * C * H * W, file);
    freadCheck(out, sizeof(float), B * C * H*2 * W*2, file);
    freadCheck(dout, sizeof(float), B * C * H*2 * W*2, file);
    freadCheck(dx, sizeof(float), B * C * H * W, file);
    fclose(file);


    // allocate device memory
    float *d_x, *d_out, *d_dout, *d_dx;
    cudaCheck(cudaMalloc(&d_x, B * C * H * W * sizeof(float)));
    cudaCheck(cudaMalloc(&d_out, B * C * H*2 * W*2 * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dout, B * C * H*2 * W*2 * sizeof(float)));
    cudaCheck(cudaMalloc(&d_dx, B * C * H * W * sizeof(float)));
    cudaCheck(cudaMemcpy(d_x, x, B * C * H * W * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_dout, dout, B * C * H*2 * W*2 * sizeof(float), cudaMemcpyHostToDevice));

    // read kernel number from command line
    int kernel_num = 1;
    if (argc > 1) {
        kernel_num = atoi(argv[1]);
    }
    printf("Running kernel %d\n", kernel_num);

    int block_sizes[] = {32, 64, 128};
    printf("Checking forward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);
        upsample_forward(kernel_num, d_out, d_x, B, C, H, W, block_size);
        validate_result(d_out, out, "out", B * C * (H/2) * (W/2));
    }
    printf("Forward pass: all results match\n\n");

    printf("Checking backward pass\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];
        printf("Checking block size %d\n", block_size);
        upsample_backward(kernel_num, d_dx, d_dout, B, C, H, W, block_size);
        validate_result(d_dx, dx, "dx", B * C * H * W);
    }
    printf("Backward pass: all results match\n\n");
    printf("All results match. Starting benchmarks.\n\n");
    printf("Forward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, upsample_forward,
                                              kernel_num, d_out, d_x,
                                              B, C, H, W, block_size);

        float tflops = (float)B * H * W * C / elapsed_time * 1e3f / 1e12f;
        printf("block_size %4d | time %.4f ms | tflops %.2f\n", block_size, elapsed_time, tflops);
    }

    printf("\nBackward pass benchmarks:\n");
    for (int j = 0; j < sizeof(block_sizes) / sizeof(int); j++) {
        int block_size = block_sizes[j];

        int repeat_times = 100;
        float elapsed_time = benchmark_kernel(repeat_times, upsample_backward,
                                              kernel_num, d_dx, d_dout,
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